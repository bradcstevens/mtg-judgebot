import os
import logging
from typing import List, Dict, Any
from data_processor import process_cards_data, process_rules_and_glossary_data
from mtg_cards_api import setup_card_database, fetch_card_details_by_oracle_id
from embeddings import initialize_embeddings
from vector_store import create_vector_store, load_vector_store, create_retriever
from langchain_core.runnables import RunnablePassthrough
from config import load_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_or_load_sqlite_db(database_path: str, cards_file_path: str, rulings_file_path: str):
    if not os.path.exists('db/mtg_cards.sqlite'):
        logger.info(f"Creating new SQLite database at {database_path}")
        setup_card_database(database_path)
        process_cards_data(cards_file_path, rulings_file_path, database_path)
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

def process_query(query: str, mtg_chain, database_path: str):
    print(f"\n{'='*50}\nProcessing query: {query}\n{'='*50}")
    try:
        retrieved_items = mtg_chain.invoke(query)
        print_search_results(retrieved_items, database_path)
    except Exception as e:
        logger.error(f"Error processing query '{query}': {e}")

def print_search_results(results, database_path: str):
    print("Retrieved Items:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Content: {result.page_content}")
        if result.metadata.get('document_type') == 'card':
            oracle_id = result.metadata['oracle_id']
            card_details = fetch_card_details_by_oracle_id(database_path, oracle_id)
            print(f"   Oracle Text: {card_details.get('oracle_text', 'N/A')}")
            print(f"   Rulings:")
            for ruling in card_details.get('rulings', []):
                print(f"     - {ruling['published_at']}: {ruling['comment']}")
        print()

def main():
    cards_file_path = 'data/oracle-cards-20240722210341.json'
    rulings_file_path = 'data/rulings-20240722210039.json'
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

    # cards_vector_store = create_or_load_vector_store(
    #     cards_vector_store_path, 
    #     embeddings, 
    #     lambda x, y: process_cards_data(x, y, database_path),
    #     [cards_file_path, rulings_file_path]
    # )

    rules_glossary_vector_store = create_or_load_vector_store(
        rules_glossary_vector_store_path,
        embeddings,
        process_rules_and_glossary_data,
        [rules_file_path, glossary_file_path]
    )

    # cards_retriever = create_retriever(cards_vector_store)
    rules_glossary_retriever = create_retriever(rules_glossary_vector_store)

    # mtg_chain = RunnablePassthrough() | (cards_retriever, rules_glossary_retriever)

    queries = [
        "What does Ogre Arsonist do?",
        "Tell me about cards that interact with Food tokens",
        "How does deathtouch interact with trample?",
    ]

    # for query in queries:
    #     process_query(query, mtg_chain, database_path)

if __name__ == "__main__":
    main()