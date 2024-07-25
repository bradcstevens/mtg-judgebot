# main.py
from card_processor import process_cards
from embeddings import initialize_embeddings
from vector_store import create_vector_store, perform_similarity_search
from config import load_api_key
from pprint import pprint

def print_search_results(query, results):
    print(f"\nSearch query: {query}")
    print("Top 3 results:")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. Content: {result.page_content[:500]}...")  # Truncate long content
        print(f"   Metadata: {result.metadata}\n")

def main():
    cards_file_path = './data/oracle-cards-20240722210341.json'
    rulings_file_path = './data/rulings-20240722210039.json'

    # Process and combine data
    combined_data = process_cards(cards_file_path, rulings_file_path)
    print(f"Total combined entries: {len(combined_data)}")
    print("Sample combined entry:", combined_data[0])

    # Load API key and initialize embeddings
    try:
        api_key = load_api_key()
        embeddings = initialize_embeddings(api_key)
    except ValueError as e:
        print(f"Error loading API key or initializing embeddings: {e}")
        return

    # Create vector store
    try:
        vector_store = create_vector_store(combined_data, embeddings)
        print("Successfully created vector store for combined data")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        import traceback
        traceback.print_exc()
        return

    # Example similarity searches
    queries = [
        "What are the rulings for first strike?",
        "How does deathtouch interact with other abilities?",
        "Are there any specific rulings for the card 'Black Lotus'?",
        "What are the rulings on planeswalker loyalty abilities?",
        "How do replacement effects work in combat?",
        "What are the rulings on casting spells for alternative costs?",
        "How do continuous effects from static abilities interact?"
    ]

    for query in queries:
        try:
            results = perform_similarity_search(vector_store, query)
            print_search_results(query, results)
        except Exception as e:
            print(f"Error performing similarity search for '{query}': {e}")

if __name__ == "__main__":
    main()