from typing import List, Dict, Any
import logging
from card_processor import process_cards_and_rulings
from rules_processor import process_rules
from glossary_processor import process_glossary
from mtg_cards_api import setup_card_database, insert_card_into_db, insert_ruling_into_db

logger = logging.getLogger(__name__)

def process_cards_for_database(cards_file_path: str, rulings_file_path: str, database_path: str):
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
                        'object': 'ruling',
                        'source': 'scryfall',
                        'published_at': '',
                        'comment': ruling
                    })
        
        logger.info(f"Processed {len(combined_data)} cards and their rulings for database")
    except Exception as e:
        logger.error(f"Error processing cards and rulings for database: {e}")
        raise

def prepare_cards_for_vector_store(cards_file_path: str, rulings_file_path: str) -> List[Dict[str, Any]]:
    try:
        combined_data = process_cards_and_rulings(cards_file_path, rulings_file_path)
        
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
        
        logger.info(f"Prepared {len(processed_cards)} cards for vector store")
        return processed_cards
    except Exception as e:
        logger.error(f"Error preparing cards for vector store: {e}")
        raise

def process_rules_data(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed_rules = []
    for rule in rules:
        try:
            content = f"Rule {rule['rule_number']}: {rule['content']}"
            metadata = {
                'id': rule['id'],
                'rule_number': rule['rule_number'],
                'base_rule': rule['base_rule'],
                'parent_rule': rule['parent_rule'],
                'document_type': 'rule'
            }
            processed_rules.append({
                'content': content,
                'metadata': metadata
            })
        except KeyError as e:
            logger.warning(f"Skipping rule due to missing key: {e}")
    logger.info(f"Processed {len(processed_rules)} rules")
    return processed_rules

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