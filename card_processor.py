from typing import List, Dict, Any
from oracle_text_processor import process_oracle_text
from ruling_processor import process_rulings

def combine_cards_and_rulings(cards: List[Dict[str, Any]], rulings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Create a dictionary to store rulings by oracle_id
    rulings_by_oracle_id = {}
    for ruling in rulings:
        oracle_id = ruling['metadata'].get('oracle_id')
        if oracle_id:
            if oracle_id not in rulings_by_oracle_id:
                rulings_by_oracle_id[oracle_id] = []
            rulings_by_oracle_id[oracle_id].append(ruling['content'])

    # Combine cards with their rulings
    combined_data = []
    for card in cards:
        oracle_id = card['content']
        card_rulings = rulings_by_oracle_id.get(oracle_id, [])
        
        combined_content = f"Card: {card['metadata'].get('name', '')}\nOracle Text: {card['metadata'].get('oracle_text', '')}"
        if card_rulings:
            combined_content += "\nRulings:\n" + "\n".join(card_rulings)
        
        combined_metadata = card['metadata'].copy()
        combined_metadata['has_rulings'] = bool(card_rulings)
        combined_metadata['ruling_count'] = len(card_rulings)
        combined_metadata['document_type'] = 'card'
        
        combined_data.append({
            'content': combined_content,
            'metadata': combined_metadata
        })
    
    return combined_data

def process_cards_and_rulings(cards_file_path: str, rulings_file_path: str) -> List[Dict[str, Any]]:
    cards = process_oracle_text(cards_file_path)
    rulings = process_rulings(rulings_file_path)
    return combine_cards_and_rulings(cards, rulings)