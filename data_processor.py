from typing import List, Dict, Any
import logging
from card_processor import process_cards_and_rulings
from rules_processor import process_rules
from glossary_processor import process_glossary
from mtg_cards_api import setup_card_database, insert_card_into_db, insert_ruling_into_db

logger = logging.getLogger(__name__)

def process_glossary_data(glossary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed_glossary = []
    for term in glossary:
        try:
            content = f"Glossary Term: {term['term']}\nDefinition: {term['definition']}"
            metadata = {
                'id': term['id'],
                'term': term['term'],
                'rule_refs': ','.join(term['rule_refs']),
                'document_type': 'glossary'
            }
            processed_glossary.append({
                'content': content,
                'metadata': metadata
            })
        except KeyError as e:
            logger.warning(f"Skipping glossary term due to missing key: {e}")
    logger.info(f"Processed {len(processed_glossary)} glossary terms")
    return processed_glossary

def process_cards_data(cards_file_path: str, rulings_file_path: str, database_path: str) -> List[Dict[str, Any]]:
    try:
        combined_data = process_cards_and_rulings(cards_file_path, rulings_file_path)
        
        setup_card_database(database_path)
        for card in combined_data:
            insert_card_into_db(database_path, card['metadata'])
            if card['metadata'].get('has_rulings'):
                rulings = card['content'].split('Rulings:\n')[1].split('\n')
                for ruling in rulings:
                    insert_ruling_into_db(database_path, {
                        'oracle_id': card['metadata']['oracle_id'],
                        'object': 'ruling',  # Add this line
                        'source': 'scryfall',  # Add a default source
                        'published_at': '',  # Add an empty published_at field
                        'comment': ruling
                    })

        # Return data for vector store (just names and oracle_ids)
        processed_cards = [
            {
                'content': card['metadata']['name'],
                'metadata': {
                    'oracle_id': card['metadata']['oracle_id'],
                    'document_type': 'card'
                }
            }
            for card in combined_data
        ]
        
        logger.info(f"Processed {len(processed_cards)} cards and their rulings")
        return processed_cards
    except Exception as e:
        logger.error(f"Error processing cards and rulings: {e}")
        raise  # Re-raise the exception to see the full error traceback

def process_rules_and_glossary_data(rules_file_path: str, glossary_file_path: str) -> List[Dict[str, Any]]:
    combined_data = []
    
    try:
        rules = process_rules(rules_file_path)
        processed_rules = process_rules_data(rules)
        combined_data.extend(processed_rules)
        logger.info(f"Processed {len(processed_rules)} rules")
    except Exception as e:
        logger.error(f"Error processing rules: {e}")
    
    try:
        glossary = process_glossary(glossary_file_path)
        processed_glossary = process_glossary_data(glossary)
        combined_data.extend(processed_glossary)
        logger.info(f"Processed {len(processed_glossary)} glossary terms")
    except Exception as e:
        logger.error(f"Error processing glossary: {e}")
    
    logger.info(f"Total combined entries for rules and glossary: {len(combined_data)}")
    return combined_data