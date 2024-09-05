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
                filtered_cards = []
                for card in card_details:
                    if card.get('oracle_text'):
                        filtered_card = {
                            'name': card.get('name'),
                            'cmc': card.get('cmc'),
                            'oracle_text': card.get('oracle_text'),
                            'legalities': card.get('legalities'),
                            'released_at': card.get('released_at'),
                            'mana_cost': card.get('mana_cost'),
                            'power': card.get('power'),
                            'toughness': card.get('toughness'),
                            'colors': card.get('colors'),
                            'color_identity': card.get('color_identity'),
                            'set_name': card.get('set_name'),
                            'rulings': [{'comment': ruling['comment']} for ruling in card.get('rulings', [])]
                        }
                        filtered_cards.append(filtered_card)
                recognized_cards.extend(filtered_cards)
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
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        )
        chunks = text_splitter.split_text(query)

        all_results = []
        for chunk in chunks:
            results = rules_vector_store.similarity_search(chunk, k=2)
            all_results.extend(results)

        unique_results = list({r.page_content: r for r in all_results}.values())

        formatted_results = []
        for i, doc in enumerate(unique_results, 1):
            formatted_results.append(f"Rule {i}:\n{doc.page_content}\n")

        return "\n".join(formatted_results)

    return StructuredTool.from_function(
        func=retrieve_relevant_rules,
        name="retrieve_relevant_rules",
        description="Retrieve relevant Magic: The Gathering rules based on the given user query or oracle text of a card or rulings of a card.",
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
        SystemMessage(content="""You are a Magic: The Gathering Judge who is answering rules questions from users.
        
        For any part of the user's query that you're unsure about or need more information on, use the retrieve_relevant_rules tool to get relevant rules information. This tool will help you provide accurate and comprehensive answers.
        
        Always use the recognize_card_names tool first to identify any card names in the query, then use the retrieve_relevant_rules tool to get relevant rules for the situation described in the query.
        You may need to use the retrieve_relevant_rules tool on the user's query, or on the oracle text or rulings from the card_names tool, if provided.
        After gathering all necessary information, provide a clear and concise answer to the user's question."""),
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
        # "I cast Michael, and my opponent responds with Hulk Smash",
        # "I attack with Satya, Aetherflux Genius. My opponent's Satya, Masterful Overlord makes 2 copies of itself. What happens?",
        # "I cast Satya, Aetherflux Genius. My opponent counters Satya. How much energy do I have?",
        "I have 4 energy. I attack my opponent with Satya, making a copy of razorfield ripper, which enters the battlefield attacking. When razorfield ripper attacks, I get an energy. When I do, my opponent casts Swords to Plowshares on my Satya. How much energy do I have now?"
    ]

    for query in example_queries:
        print(f"\n{'='*50}\nProcessing query: {query}\n{'='*50}")
        result = agent_executor.invoke({"input": query})
        print(f"Agent response: {result['output']}")

if __name__ == "__main__":
    main()