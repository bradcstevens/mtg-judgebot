import os
import logging
from data_processor import process_all_data
from embeddings import initialize_embeddings
from vector_store import (
    create_vector_store, 
    load_vector_store, 
    create_retriever, 
    perform_similarity_search,
)
from retrieval_chain import create_retrieval_chain
from config import load_api_key

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_search_results(query, results):
    print(f"\nSearch query: {query}")
    print("Top 3 results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Content: {result.page_content}")
        print(f"   Metadata: {result.metadata}\n")

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

    persist_directory = "./chroma_db"

    # Check if vector store exists
    if os.path.exists(persist_directory):
        logger.info("Loading existing vector store...")
        vector_store = load_vector_store(persist_directory, embeddings)
    else:
        logger.info("Vector store not found. Creating new vector store...")
        try:
            combined_data = process_all_data(cards_file_path, rulings_file_path, rules_file_path, glossary_file_path)
            logger.info(f"Total combined entries: {len(combined_data)}")
            if len(combined_data) == 0:
                logger.error("No data was processed successfully.")
                return
            vector_store = create_vector_store(combined_data, embeddings, persist_directory)
        except Exception as e:
            logger.error(f"Error processing data or creating vector store: {e}")
            return

    logger.info(f"Number of documents in vector store: {len(vector_store.get())}")

    # Create retriever
    retriever = create_retriever(vector_store)

    # Create retrieval chain
    try:
        mtg_chain = create_retrieval_chain(retriever, api_key)
    except Exception as e:
        logger.error(f"Error creating retrieval chain: {e}")
        return

    # Example queries
    queries = [
        "I have 6 energy and attack with Satya. I make a copy of RazorfieldRipper. How much energy do I have now?",
        # "How does deathtouch interact with trample?",
        # "Explain the rules for casting spells with alternative costs",
        # "What is the definition of 'mana' in Magic: The Gathering?",
        # "How do planeswalker loyalty abilities work?",
    ]

    for query in queries:
        print(f"\n{'='*50}\nProcessing query: {query}\n{'='*50}")
        try:
            # First, let's see what the similarity search returns
            search_results = perform_similarity_search(vector_store, query, k=20)
            print_search_results(query, search_results)

            # Now, let's see what the retrieval chain returns
            chain_response = mtg_chain.invoke(query)
            print("\nRetrieval Chain Response:")
            print(chain_response)
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")

if __name__ == "__main__":
    main()