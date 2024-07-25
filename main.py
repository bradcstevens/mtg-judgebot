from card_processor import process_cards
from ruling_processor import process_rulings
from rules_processor import process_rules
from glossary_processor import process_glossary
from embeddings import initialize_embeddings
from vector_store import create_vector_store, perform_similarity_search
from config import load_api_key
from pprint import pprint

def main():
    cards_file_path = './data/oracle-cards-20240722210341.json'
    rulings_file_path = './data/rulings-20240722210039.json'
    rules_file_path = './data/official-rules.txt'
    glossary_file_path = './data/glossary.txt'

    # Process data
    cards = process_cards(cards_file_path)
    rulings = process_rulings(rulings_file_path)
    rules = process_rules(rules_file_path)
    glossary = process_glossary(glossary_file_path)

    print(f"Total cards processed: {len(cards)}")
    print(f"Total rulings processed: {len(rulings)}")
    print(f"Total rule chunks processed: {len(rules)}")
    print(f"Total glossary chunks processed: {len(glossary)}")

    # Load API key and initialize embeddings
    try:
        api_key = load_api_key()
        embeddings = initialize_embeddings(api_key)
    except ValueError as e:
        print(f"Error loading API key or initializing embeddings: {e}")
        return

    # Create vector stores
    vector_stores = {}
    for data_type, data in [('cards', cards), ('rulings', rulings), ('rules', rules), ('glossary', glossary)]:
        try:
            vector_stores[data_type] = create_vector_store(data, embeddings, data_type)
            print(f"Successfully created vector store for {data_type}")
        except Exception as e:
            print(f"Error creating vector store for {data_type}: {e}")

    if not vector_stores:
        print("No vector stores were successfully created. Exiting.")
        return

    # Example similarity search
    query = "What is the rule for first strike?"
    for data_type, db in vector_stores.items():
        try:
            results = perform_similarity_search(db, query)
            print(f"\nSearch results from {data_type}:")
            print(results[0].page_content if results else "No results")
        except Exception as e:
            print(f"Error performing similarity search on {data_type}: {e}")

if __name__ == "__main__":
    main()