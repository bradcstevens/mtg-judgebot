from pprint import pprint
from card_processor import process_cards
from ruling_processor import process_rulings

def main():
    cards_file_path = './data/oracle-cards-20240722210341.json'
    rulings_file_path = './data/rulings-20240722210039.json'

    cards = process_cards(cards_file_path)
    rulings = process_rulings(rulings_file_path)

    # Print the first few processed cards
    print("First few cards:")
    pprint(cards[:2])

    # Print the first few processed rulings
    print("\nFirst few rulings:")
    pprint(rulings[:2])

    # Print total number of cards and rulings processed
    print(f"\nTotal cards processed: {len(cards)}")
    print(f"Total rulings processed: {len(rulings)}")

if __name__ == "__main__":
    main()