from data_processor import process_all_data
from embeddings import initialize_embeddings
from vector_store import create_vector_store, load_vector_store, create_retriever, perform_similarity_search
from retrieval_chain import create_retrieval_chain
from config import load_api_key
import os

def print_search_results(query, results):
    print(f"\nSearch query: {query}")
    print("Top 3 results:")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. Content: {result.page_content[:200]}...")  # Truncate to first 200 characters
        print(f"   Metadata: {result.metadata}\n")

def main():
    cards_file_path = './data/oracle-cards-20240722210341.json'
    rulings_file_path = './data/rulings-20240722210039.json'
    rules_file_path = './data/official-rules.txt'
    glossary_file_path = './data/glossary.txt'

    # Process all data
    combined_data = process_all_data(cards_file_path, rulings_file_path, rules_file_path, glossary_file_path)
    print(f"Total combined entries: {len(combined_data)}")

    # Load API key and initialize embeddings
    try:
        api_key = load_api_key()
        embeddings = initialize_embeddings(api_key)
    except ValueError as e:
        print(f"Error loading API key or initializing embeddings: {e}")
        return

    persist_directory = "./chroma_db"

    # Create or load vector store
    if os.path.exists(persist_directory):
        print("Loading existing vector store...")
        vector_store = load_vector_store(persist_directory, embeddings)
    else:
        print("Creating new vector store...")
        vector_store = create_vector_store(combined_data, embeddings, persist_directory)
        print("Vector store created and persisted.")

    # Create retriever
    retriever = create_retriever(vector_store)

    # Create retrieval chain
    mtg_chain = create_retrieval_chain(retriever, api_key)

    # Example queries
    queries = [
        "What is the commander tax rule?",
        "How does deathtouch interact with trample?",
        "Explain the rules for casting spells with alternative costs",
        "What is the definition of 'mana' in Magic: The Gathering?",
        "How do planeswalker loyalty abilities work?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        try:
            # First, let's see what the similarity search returns
            search_results = perform_similarity_search(vector_store, query, k=3)
            print("Similarity Search Results:")
            for i, result in enumerate(search_results, 1):
                print(f"{i}. Content: {result.page_content[:200]}...")  # Print first 200 characters of each result
                print(f"   Metadata: {result.metadata}\n")

            # Now, let's see what the retrieval chain returns
            chain_response = mtg_chain.invoke(query)
            print("\nRetrieval Chain Response:")
            print(chain_response)
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()