import os
import logging
from typing import List
import torch
import json
from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from mtg_cards_api import fetch_card_by_name
from data_processor import process_cards_for_database, prepare_cards_for_vector_store, process_rules_and_glossary_data
from mtg_cards_api import fetch_card_details_by_oracle_id
from embeddings import initialize_embeddings
from vector_store import create_vector_store, load_vector_store, create_retriever
from config import load_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

database_path = "db/mtg_cards.sqlite"

class CardNameRecognitionInput(BaseModel):
    card_names: List[str] = Field(..., description="List of Magic: The Gathering card names to analyze")

class RulesRetrievalInput(BaseModel):
    query: str = Field(..., description="The query or context to search for relevant rules")

def create_card_name_recognition_tool():
    def recognize_card_names(card_names):
        recognized_cards = []
        for card_name in card_names:
            card_details = fetch_card_by_name(database_path, card_name)
            if card_details:
                recognized_cards.extend(card_details)
            else:
                logger.warning(f"Card not found: {card_name}")
        
        logger.info(f"Recognized cards: {[card['name'] for card in recognized_cards]}")
        return json.dumps(recognized_cards, indent=2)

    return StructuredTool.from_function(
        func=recognize_card_names,
        name="recognize_card_names",
        description="Recognize and log Magic: The Gathering card names from the user's input. Log everything that could conceivably be a card name. This includes card names you do not know. Your criteria for deciding whether to include it is that it is used in the sentence in a way that a Magic the Gathering card name might be. Your goal is to retrieve a unique list of card names used in the user's question so we can look up more information about those card names.",
        args_schema=CardNameRecognitionInput
    )

def create_rules_retrieval_tool(rules_vector_store):
    def retrieve_relevant_rules(query: str) -> str:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_text(query)

        all_results = []
        for chunk in chunks:
            results = rules_vector_store.similarity_search_with_score(
                chunk, k=3
            )
            logging.debug(f"Raw results: {results}")
            
            for doc, score in results:
                doc.metadata['relevance_score'] = 1.0 - score  # Convert distance to similarity
                all_results.append(doc)

            logging.info(f"Results with scores: {[(r.page_content[:50], r.metadata.get('relevance_score')) for r in all_results[-3:]]}")

        unique_results = list({r.page_content: r for r in all_results}.values())

        logging.info(f"Total unique results before filtering: {len(unique_results)}")
        for i, r in enumerate(unique_results):
            logging.info(f"Result {i+1} metadata: {r.metadata}")
            score = r.metadata.get('relevance_score', 'N/A')
            logging.info(f"Result {i+1} score: {score if score == 'N/A' else f'{score:.4f}'}")

        threshold = 0.75  # Adjust this value based on observed scores
        filtered_results = [r for r in unique_results if r.metadata.get('relevance_score', 0) >= threshold]

        logging.info(f"Results after filtering (threshold {threshold}): {len(filtered_results)}")
        for i, r in enumerate(filtered_results):
            score = r.metadata.get('relevance_score', 'N/A')
            logging.info(f"Filtered result {i+1} metadata: {r.metadata}")
            logging.info(f"Filtered result {i+1} score: {score if score == 'N/A' else f'{score:.4f}'}")

        formatted_results = []
        for i, doc in enumerate(filtered_results, 1):
            score = doc.metadata.get('relevance_score', 'N/A')
            formatted_results.append(f"Rule {i} (Score: {score if score == 'N/A' else f'{score:.4f}'}):\n{doc.page_content}\n")

        return "\n".join(formatted_results) if formatted_results else "No relevant rules found."

    return StructuredTool.from_function(
        func=retrieve_relevant_rules,
        name="retrieve_relevant_rules",
        description="Retrieve relevant Magic: The Gathering rules based on the given query or context. Use this tool whenever there is an interaction, game state, rule, or card text or ruling that you have any degree of ambiguity about. To use this tool, replace card names with relevant text you have retrieved about them from oracle_text or rulings. "
        "For example, replace a creature's name with creature or its subtypes. Replace a spell like murder with its effect of destroy a creature.",
        args_schema=RulesRetrievalInput
    )

# Load the trained model and tokenizer
model_path = "models/mtg_card_name_model"
tokenizer_path = "models/tokenizer"
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

def create_or_load_sqlite_db(database_path: str, cards_file_path: str, rulings_file_path: str):
    if not os.path.exists(database_path):
        logger.info(f"Creating new SQLite database at {database_path}")
        process_cards_for_database(cards_file_path, rulings_file_path, database_path)
    else:
        logger.info(f"SQLite database already exists at {database_path}")

def create_or_load_vector_store(persist_directory: str, embeddings, data_processor, file_paths: List[str]):
    if os.path.exists(persist_directory):
        logger.info(f"Loading existing vector store from {persist_directory}")
        return load_vector_store(persist_directory, embeddings)
    else:
        logger.info(f"Creating new vector store in {persist_directory}")
        data = data_processor(*file_paths)
        return create_vector_store(data, embeddings, persist_directory)

def print_search_results(results):
    print("Retrieved Cards:")
    for i, result in enumerate(results, 1):
        if result.metadata.get('document_type') == 'card':
            print(f"{i}. Card Name: {result.page_content}")
            print(f"   Oracle ID: {result.metadata['oracle_id']}")
        else:
            print(f"{i}. Content: {result.page_content}")
        print()

def create_react_agent(llm, tools):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
1. Role:
You are a Magic: The Gathering Judge who is answering rules questions from users. Remember that text in Magic: The Gathering is like a code or contract. Take words as precisely what they mean. You will not understand the words colloquially but be hyper-critical in your analysis. The words mean exactly what they say.

For example, if a creature says that when it attacks, I get 2 life, but the creature is created already tapped and attacking, I don't get two life, because it never "became attacking" by declaring an attack in the declare attackers step, which is what would be required to get the trigger to get 2 life.

2. Using Tools:
a) Card Name Recognition Tool:
If you believe a query has card names in it, use the recognize_card_names tool first to retrieve card text and rulings.

b) Retrieval Rules Tool:
For any part of the user's query that you're unsure about or need more information on, use the retrieve_relevant_rules tool to get relevant rules information. This tool will help you provide accurate and comprehensive answers.

Then, break down the query into smaller, specific rules questions. Use the retrieve_relevant_rules tool multiple times, once for each specific rules question or game mechanic you need to clarify. For example, you might make separate calls for different card abilities, interactions, or game states mentioned in the query.
To use the retrieval tool, first look at the glossary. This is a glossary whose full text you will be searching with a vector search. Think step by step to decide which rules will be helpful to answer the question and create queries that will look them up.

Use the retrieve_relevant_rules tool without card names, just the text of the rule, or the type of the card, for example "Whenever creature attacks" instead of "whever Satya attacks".

Glossary:
1. Game Concepts
100. General
101. The Magic Golden Rules
102. Players
103. Starting the Game
104. Ending the Game
105. Colors
106. Mana
107. Numbers and Symbols
108. Cards
109. Objects
110. Permanents
111. Tokens
112. Spells
113. Abilities
114. Emblems
115. Targets
116. Special Actions
117. Timing and Priority
118. Costs
119. Life
120. Damage
121. Drawing a Card
122. Counters
123. Stickers

2. Parts of a Card
200. General
201. Name
202. Mana Cost and Color
203. Illustration
204. Color Indicator
205. Type Line
206. Expansion Symbol
207. Text Box
208. Power/Toughness
209. Loyalty
210. Defense
211. Hand Modifier
212. Life Modifier
213. Information Below the Text Box

3. Card Types
300. General
301. Artifacts
302. Creatures
303. Enchantments
304. Instants
305. Lands
306. Planeswalkers
307. Sorceries
308. Kindreds
309. Dungeons
310. Battles
311. Planes
312. Phenomena
313. Vanguards
314. Schemes
315. Conspiracies

4. Zones
400. General
401. Library
402. Hand
403. Battlefield
404. Graveyard
405. Stack
406. Exile
407. Ante
408. Command

5. Turn Structure
500. General
501. Beginning Phase
502. Untap Step
503. Upkeep Step
504. Draw Step
505. Main Phase
506. Combat Phase
507. Beginning of Combat Step
508. Declare Attackers Step
509. Declare Blockers Step
510. Combat Damage Step
511. End of Combat Step
512. Ending Phase
513. End Step
514. Cleanup Step

6. Spells, Abilities, and Effects
600. General
601. Casting Spells
602. Activating Activated Abilities
603. Handling Triggered Abilities
604. Handling Static Abilities
605. Mana Abilities
606. Loyalty Abilities
607. Linked Abilities
608. Resolving Spells and Abilities
609. Effects
610. One-Shot Effects
611. Continuous Effects
612. Text-Changing Effects
613. Interaction of Continuous Effects
614. Replacement Effects
615. Prevention Effects
616. Interaction of Replacement and/or Prevention Effects

7. Additional Rules
700. General
701. Keyword Actions
702. Keyword Abilities
703. Turn-Based Actions
704. State-Based Actions
705. Flipping a Coin
706. Rolling a Die
707. Copying Objects
708. Face-Down Spells and Permanents
709. Split Cards
710. Flip Cards
711. Leveler Cards
712. Double-Faced Cards
713. Substitute Cards
714. Saga Cards
715. Adventurer Cards
716. Class Cards
717. Attraction Cards
718. Prototype Cards
719. Case Cards
720. Controlling Another Player
721. Ending Turns and Phases
722. The Monarch
723. The Initiative
724. Restarting the Game
725. Rad Counters
726. Subgames
727. Merging with Permanents
728. Day and Night
729. Taking Shortcuts
730. Handling Illegal Actions

8. Multiplayer Rules
800. General
801. Limited Range of Influence Option
802. Attack Multiple Players Option
803. Attack Left and Attack Right Options
804. Deploy Creatures Option
805. Shared Team Turns Option
806. Free-for-All Variant
807. Grand Melee Variant
808. Team vs. Team Variant
809. Emperor Variant
810. Two-Headed Giant Variant
811. Alternating Teams Variant

9. Casual Variants
900. General
901. Planechase
902. Vanguard
903. Commander
904. Archenemy
905. Conspiracy Draft

3. The Magic Golden Rules:
101.1. Whenever a card's text directly contradicts these rules, the card takes precedence. The card overrides only the rule that applies to that specific situation. The only exception is that a player can concede the game at any time (see rule 104.3a).

101.2. When a rule or effect allows or directs something to happen, and another effect states that it can't happen, the "can't" effect takes precedence.
Example: If one effect reads "You may play an additional land this turn" and another reads "You can't play lands this turn," the effect that precludes you from playing lands wins.

101.2a Adding abilities to objects and removing abilities from objects don't fall under this rule. (See rule 113.10.)

101.3. Any part of an instruction that's impossible to perform is ignored. (In many cases the card will specify consequences for this; if it doesn't, there's no effect.)

101.4. If multiple players would make choices and/or take actions at the same time, the active player (the player whose turn it is) makes any choices required, then the next player in turn order (usually the player seated to the active player's left) makes any choices required, followed by the remaining nonactive players in turn order. Then the actions happen simultaneously. This rule is often referred to as the "Active Player, Nonactive Player (APNAP) order" rule.
Example: A card reads "Each player sacrifices a creature." First, the active player chooses a creature they control. Then each of the nonactive players, in turn order, chooses a creature they control. Then all creatures chosen this way are sacrificed simultaneously.

101.4a If an effect has each player choose a card in a hidden zone, such as their hand or library, those cards may remain face down as they're chosen. However, each player must clearly indicate which face-down card they are choosing.

101.4b A player knows the choices made by the previous players when making their choice, except as specified in 101.4a.

101.4c If a player would make more than one choice at the same time, the player makes the choices in the order specified. If no order is specified, the player chooses the order.

101.4d If a choice made by a nonactive player causes the active player, or a different nonactive player earlier in the turn order, to have to make a choice, APNAP order is restarted for all outstanding choices.

101.4e If multiple players would make choices or take actions while starting the game, the starting player is considered the active player and each other player is considered a nonactive player.

4. Answering User's Query:
Think step by step when evaluataing final input aand remember that Magic:the Gathering is a game where exact wording and technicalities matter, so be precise in your evaluation.

After gathering all necessary information, provide a clear and concise answer to the user's question.

Remember, you are a judge, so defer to the rules to make the decision. The person asking the question is only a player and may have given you bad information based on their flawed understanding. Use the rules and tools you have to answer the question.
"""),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = OpenAIFunctionsAgent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        return_intermediate_steps=True
    )

def main():
    cards_file_path = 'data/oracle-cards-20240722210341.json'
    rulings_file_path = 'data/rulings-20240901210034.json'
    rules_file_path = 'data/official-rules.txt'
    glossary_file_path = 'data/glossary.txt'
    
    database_path = "db/mtg_cards.sqlite"
    cards_vector_store_path = "db/chroma_db_cards"
    rules_glossary_vector_store_path = "db/chroma_db_rules_glossary"

    file_paths = [cards_file_path, rulings_file_path, rules_file_path, glossary_file_path]
    if not all(os.path.exists(path) for path in file_paths):
        logger.error("One or more required files not found.")
        return

    try:
        api_key = load_api_key()
        embeddings = initialize_embeddings(api_key)
    except ValueError as e:
        logger.error(f"Error loading API key or initializing embeddings: {e}")
        return

    create_or_load_sqlite_db(database_path, cards_file_path, rulings_file_path)

    cards_vector_store = create_or_load_vector_store(
        cards_vector_store_path, 
        embeddings, 
        prepare_cards_for_vector_store,
        [cards_file_path, rulings_file_path]
    )

    rules_glossary_vector_store = create_or_load_vector_store(
        rules_glossary_vector_store_path, 
        embeddings, 
        process_rules_and_glossary_data,
        [rules_file_path, glossary_file_path]
    )

    # Create the tools
    card_name_tool = create_card_name_recognition_tool()
    rules_retrieval_tool = create_rules_retrieval_tool(rules_glossary_vector_store)

    # Create the agent with both tools
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    tools = [card_name_tool, rules_retrieval_tool]
    agent_executor = create_react_agent(llm, tools)

    # Example queries
    example_queries = [
        "Does this sequencing work?\n"
        "[[Duke Ulder Ravenguard]], [[Taranika, Akroan Veteran]], [[Bounty Agent]] are on the board. Combat happens, and Duke will trigger, targeting the Bounty Agent. Myriad would trigger, creating copies of Bounty Agent attacking each opponent. Taranika would then trigger to target one of the copies of Bounty Agent. Can you then tap that copy of Bounty Agent to trigger activated ability?"
    ]

    for query in example_queries:
        print(f"\n{'='*50}\nProcessing query: {query}\n{'='*50}")
        result = agent_executor.invoke({"input": query})
        print(f"Agent response: {result['output']}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()