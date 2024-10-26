import os
import logging
from typing import List, Optional, TypedDict, Union, Sequence, Annotated
import json
from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

from mtg_cards_api import fetch_card_by_name
from data_processor import process_cards_for_database, prepare_cards_for_vector_store
from embeddings import initialize_embeddings
from vector_store import create_vector_store, load_vector_store
from config import load_api_key
from rules_api import get_rule_and_children
from app.api.chat.tools.game_state_constructor import GameStateConstructor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

database_path = "db/mtg_cards.sqlite"

class CardNameRecognitionInput(BaseModel):
    card_names: List[str] = Field(..., description="List of Magic: The Gathering card names to analyze")

class GraphState(TypedDict):
    question: str
    card_names: Optional[List[str]]
    rules: Optional[str]
    game_state: Optional[str]
    response: Optional[str]

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

class RulesLookupInput(BaseModel):
    rule_numbers: List[str] = Field(..., description="List of Magic: The Gathering rule numbers to look up")

def rules_lookup(rule_numbers: List[str]) -> str:
    responses = []
    for rule_number in rule_numbers:
        print(f"Fetching rule {rule_number} and its subrules")  # Debug print
        rule_data = get_rule_and_children(rule_number)
        print(f"Rule data for {rule_number}: {rule_data}")  # Debug print
        
        if not rule_data:
            responses.append(f"Rule {rule_number} not found.")
        else:
            response = f"Rule {rule_data['rule_number']}: {rule_data['content']}\n\n"
            
            if rule_data['children']:
                response += "Sub-rules:\n"
                for child in rule_data['children']:
                    response += f"- {child['rule_number']}: {child['content']}\n"
            else:
                response += "No sub-rules found.\n"
            
            responses.append(response)
    
    full_response = "\n\n".join(responses)
    print(f"Rules lookup full response:\n{full_response}")  # Debug print
    return full_response

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
You first assumption will be that you know nothing about the rules and card game.
To use the retrieval tool, first look at the glossary. Decide which rule you need to know to answer the question, and look it up by rule number.
For example, if a card says "Whenever creature attacks, destroy target creature", you might look up "Triggered Ability, "Targets", and "Declare Attackers Step."
Always look up the rules for targeting if you have any effects that target.
Always look up rule 405 the stack.
Always look up all rules on all phases.
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
Forget any information that has to do with the order of execution of things that the user has told you.
Reconstruct the order based on the board state they have described.
Think step by step when evaluataing final input and remember that Magic:the Gathering is a game where exact wording and technicalities matter, so be precise in your evaluation.

After gathering all necessary information, use the information to answer the question and defend your answer. It must be answered only with the information you retrieved, otherwise you will look up more information until you know the answer.

5. Using the Game State Constructor:
Before answering any question, use the game_state_constructor tool to create a detailed representation of the game state based on the user's description. Use this game state as the foundation for your reasoning and decision-making process.

### ANSWERING A QUESTION ###
YOU ARE A MAGIC: THE GATHERING RULES ENGINE.
To answer a quesstion, you must have a full understanding of the state of the game. What abilities are on the stack? What phases are the players in? What abilities have resolved? You must think step by step, like a game engine.
Decide what is the beginning of time for the purposes of the user's query. Describe your understanding of the board state at that time.
REMEMBER THERE IS A DIFFERENCE BETWEEN WHEN AN ABILITY COMES ON THE STACK AND WHEN IT RESOLVES.
AN ABILITY DOES NOT RESOLVE AT THE SAME TIME AS IT COMES ON THE STACK.
WHEN AN ABILITY TRIGGERS, YOU CAAN SAY "ABILITY TRIGGERS, PLACING AN ABILITY ON THE STACK THAT DOES X"
NOT "An ability triggers, doing X"
Abilities triggered at the same time are placed on the stack at the same time.
All targets for abilities must be chosen when the abilities are placed on the stack.
Then, describe every event that happens as a game engine would. Describe when every ability goes on the stack. Describe when an ability resolves. Describe as players move through phases.
Example: "You declare attackers, triggering 2 abilities. They both go on the stack. Ability one targets something. Ability 2 creates copies. We pick targets for triggered abilities. Now they resolve in reverse order that you chose to put them on stack.
YOU ARE NOW A MAGIC: THE GATHERING JUDGE.
As part of your output, tell me all rules you looked up and all sub-rules you used.
Print out the FULL TEXT OF ALL INFORMATION YOU USED TO MAKE YOUR DECISION.
This is the text for duke ulder ravenguard "
At the beginning of combat on your turn, another target creature you control gains haste and myriad until end of turn. (Whenever it attacks, for each opponent other than defending player, you may create a token copy that's tapped and attacking that player or a planeswalker they control. Exile the tokens at end of combat.)"
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

def card_name_recognition(state: GraphState) -> GraphState:
    card_name_tool = create_card_name_recognition_tool()
    result = card_name_tool.run(state["question"])
    state["card_names"] = json.loads(result)
    return state

def rules_lookup(state: GraphState) -> GraphState:
    rules_lookup_tool = StructuredTool.from_function(
        func=rules_lookup,
        name="rules_lookup",
        description="Look up multiple Magic: The Gathering rules by their numbers",
        args_schema=RulesLookupInput
    )
    # For simplicity, let's assume we're looking up rule 100
    result = rules_lookup_tool.run(["100"])
    state["rules"] = result
    return state

def game_state_construction(state: GraphState) -> GraphState:
    game_state_constructor = GameStateConstructor()
    result = game_state_constructor.run(state["question"])
    state["game_state"] = result
    return state

def agent_execution(state: GraphState) -> Union[GraphState, Sequence[Annotated[GraphState, "final_answer"]]]:
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    tools = [
        create_card_name_recognition_tool(),
        StructuredTool.from_function(
            func=rules_lookup,
            name="rules_lookup",
            description="Look up multiple Magic: The Gathering rules by their numbers",
            args_schema=RulesLookupInput
        ),
        Tool(
            name="game_state_constructor",
            func=GameStateConstructor().run,
            description=GameStateConstructor().description
        )
    ]
    agent_executor = create_react_agent(llm, tools)
    
    result = agent_executor.invoke({
        "input": state["question"],
        "card_names": state["card_names"],
        "rules": state["rules"],
        "game_state": state["game_state"]
    })
    
    state["response"] = result["output"]
    return [state]

def main():
    cards_file_path = 'data/oracle-cards-20240722210341.json'
    rulings_file_path = 'data/rulings-20240901210034.json'
    
    database_path = "db/mtg_cards.sqlite"
    cards_vector_store_path = "db/chroma_db_cards"

    file_paths = [cards_file_path, rulings_file_path]
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

    workflow = StateGraph(GraphState)

    workflow.add_node("card_name_recognition", card_name_recognition)
    workflow.add_node("rules_lookup", rules_lookup)
    workflow.add_node("game_state_construction", game_state_construction)
    workflow.add_node("agent_execution", agent_execution)

    workflow.set_entry_point("card_name_recognition")
    workflow.add_edge("card_name_recognition", "rules_lookup")
    workflow.add_edge("rules_lookup", "game_state_construction")
    workflow.add_edge("game_state_construction", "agent_execution")
    workflow.add_edge("agent_execution", END)

    app = workflow.compile()

    # Example queries
    example_queries = [
        "Does this sequencing work?\n"
        "[[Duke Ulder Ravenguard]], [[Taranika, Akroan Veteran]], [[Bounty Agent]] are on the board. Combat happens, and Duke will trigger, targeting the Bounty Agent. Myriad would trigger, creating copies of Bounty Agent attacking each opponent. Taranika would then trigger to target one of the copies of Bounty Agent. Can you then tap that copy of Bounty Agent to trigger activated ability?"
    ]

    for query in example_queries:
        print(f"\n{'='*50}\nProcessing query: {query}\n{'='*50}")
        result = app.invoke({"question": query})
        print(f"Agent response: {result['response']}")

if __name__ == "__main__":
    main()
