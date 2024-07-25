from data_processor import process_all_data
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
    rules_file_path = './data/official-rules.txt'
    glossary_file_path = './data/glossary.txt'

    # Process all data
    combined_data = process_all_data(cards_file_path, rulings_file_path, rules_file_path, glossary_file_path)
    print(f"Total combined entries: {len(combined_data)}")
    print("Sample entries:")
    for doc_type in ['card', 'rule', 'glossary']:
        sample = next((item for item in combined_data if item['metadata']['document_type'] == doc_type), None)
        if sample:
            print(f"\nSample {doc_type}:")
            pprint(sample)

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
        "Explain the rules for casting spells with alternative costs",
        "What is the definition of 'mana' in Magic: The Gathering?",
        "How do planeswalker loyalty abilities work according to the rules?",
        "What are the specific rules for the 'commander' format?",
        "Explain the concept of 'state-based actions' in Magic: The Gathering",
    ]

    for query in queries:
        try:
            results = perform_similarity_search(vector_store, query)
            print_search_results(query, results)
        except Exception as e:
            print(f"Error performing similarity search for '{query}': {e}")

if __name__ == "__main__":
    main()