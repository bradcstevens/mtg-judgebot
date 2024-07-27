from typing import List, Dict, Any
import logging
from card_processor import process_cards_and_rulings
from rules_processor import process_rules
from glossary_processor import process_glossary

logger = logging.getLogger(__name__)

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

def process_cards_data(cards_file_path: str, rulings_file_path: str) -> List[Dict[str, Any]]:
    try:
        cards_and_rulings = process_cards_and_rulings(cards_file_path, rulings_file_path)
        logger.info(f"Processed {len(cards_and_rulings)} cards and rulings")
        return cards_and_rulings
    except Exception as e:
        logger.error(f"Error processing cards and rulings: {e}")
        return []

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

def process_all_data(cards_file_path: str, rulings_file_path: str, rules_file_path: str, glossary_file_path: str) -> List[Dict[str, Any]]:
    combined_data = []
    combined_data.extend(process_cards_data(cards_file_path, rulings_file_path))
    combined_data.extend(process_rules_and_glossary_data(rules_file_path, glossary_file_path))
    logger.info(f"Total combined entries: {len(combined_data)}")
    return combined_data