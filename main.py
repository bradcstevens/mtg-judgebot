import os
import logging
from data_processor import process_cards_data, process_rules_and_glossary_data
from embeddings import initialize_embeddings
from vector_store import (
    create_vector_store, 
    load_vector_store, 
    create_retriever, 
    perform_similarity_search,
)
from langchain_core.runnables import RunnablePassthrough
from config import load_api_key

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_search_results(results):
    print("Retrieved Cards:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Content: {result.page_content}")
        print(f"   Metadata: {result.metadata}\n")

def create_or_load_vector_store(persist_directory, embeddings, process_data_func):
    if os.path.exists(persist_directory):
        logger.info(f"Loading existing vector store from {persist_directory}...")
        vector_store = load_vector_store(persist_directory, embeddings)
    else:
        logger.info(f"Creating new vector store in {persist_directory}...")
        try:
            data = process_data_func()
            logger.info(f"Total entries: {len(data)}")
            if len(data) == 0:
                logger.error("No data was processed successfully.")
                return None
            vector_store = create_vector_store(data, embeddings, persist_directory)
        except Exception as e:
            logger.error(f"Error processing data or creating vector store: {e}")
            return None
    
    logger.info(f"Number of documents in vector store: {len(vector_store.get())}")
    return vector_store

def create_mtg_chain(cards_retriever):
    return RunnablePassthrough() | cards_retriever

def process_query(query, mtg_chain):
    print(f"\n{'='*50}\nProcessing query: {query}\n{'='*50}")
    try:
        # Get response from the MTG chain
        retrieved_cards = mtg_chain.invoke(query)
        print_search_results(retrieved_cards)
    except Exception as e:
        logger.error(f"Error processing query '{query}': {e}")

def main():
    cards_file_path = './data/oracle-cards-20240722210341.json'
    rulings_file_path = './data/rulings-20240722210039.json'
    rules_file_path = './data/official-rules.txt'
    glossary_file_path = './data/glossary.txt'

    # Check if all files exist
    for path in [cards_file_path, rulings_file_path, rules_file_path, glossary_file_path]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return

    # Load API key and initialize embeddings
    try:
        api_key = load_api_key()
        embeddings = initialize_embeddings(api_key)
    except ValueError as e:
        logger.error(f"Error loading API key or initializing embeddings: {e}")
        return

    cards_persist_directory = "./chroma_db_cards"
    rules_persist_directory = "./chroma_db_rules"

    # Create or load card vector store
    cards_vector_store = create_or_load_vector_store(
        cards_persist_directory, 
        embeddings, 
        lambda: process_cards_data(cards_file_path, rulings_file_path)
    )

    # Create or load rules vector store
    rules_vector_store = create_or_load_vector_store(
        rules_persist_directory, 
        embeddings, 
        lambda: process_rules_and_glossary_data(rules_file_path, glossary_file_path)
    )

    # Create retriever
    cards_retriever = create_retriever(cards_vector_store)

    # Create MTG chain
    mtg_chain = create_mtg_chain(cards_retriever)

    # Example queries
    queries = [
        "I have 6 energy and attack with Satya. I make a copy of RazorfieldRipper. How much energy do I have now?",
        # "How does deathtouch interact with trample?",
        # "Explain the rules for casting spells with alternative costs",
        # "What is the definition of 'mana' in Magic: The Gathering?",
        # "How do planeswalker loyalty abilities work?",
    ]

    for query in queries:
        process_query(query, mtg_chain)

if __name__ == "__main__":
    main()