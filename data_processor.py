from typing import List, Dict, Any
import logging
from card_processor import process_cards_and_rulings
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
