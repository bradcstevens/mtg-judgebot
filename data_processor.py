from typing import List, Dict, Any
from card_processor import process_cards_and_rulings
from rules_processor import process_rules
from glossary_processor import process_glossary

def process_rules_data(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed_rules = []
    for rule in rules:
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
    return processed_rules

def process_glossary_data(glossary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed_glossary = []
    for term in glossary:
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
    return processed_glossary

def process_all_data(cards_file_path: str, rulings_file_path: str, rules_file_path: str, glossary_file_path: str) -> List[Dict[str, Any]]:
    cards_and_rulings = process_cards_and_rulings(cards_file_path, rulings_file_path)
    rules = process_rules(rules_file_path)
    glossary = process_glossary(glossary_file_path)
    
    processed_rules = process_rules_data(rules)
    processed_glossary = process_glossary_data(glossary)
    
    return cards_and_rulings + processed_rules + processed_glossary