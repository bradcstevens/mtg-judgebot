from card_processor import process_cards
from ruling_processor import process_rulings
from rules_processor import process_rules
from glossary_processor import process_glossary
from pprint import pprint

def main():
    cards_file_path = './data/oracle-cards-20240722210341.json'
    rulings_file_path = './data/rulings-20240722210039.json'
    rules_file_path = './data/official-rules.txt'
    glossary_file_path = './data/glossary.txt'

    cards = process_cards(cards_file_path)
    rulings = process_rulings(rulings_file_path)
    rules = process_rules(rules_file_path)
    glossary = process_glossary(glossary_file_path)

    print(f"Total cards processed: {len(cards)}")
    print(f"Total rulings processed: {len(rulings)}")
    print(f"Total rule chunks processed: {len(rules)}")
    print(f"Total glossary chunks processed: {len(glossary)}")

    print("\nFirst card:")
    pprint(cards[:10])

    print("\nFirst rules:")
    pprint(rules[:10])

    print("\nFirst ruling:")
    pprint(rulings[:10])

    print("\nFirst rule chunk:")
    pprint(glossary[:10])

if __name__ == "__main__":
    main()

