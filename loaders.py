import json
from typing import Dict, List, Generator
from pprint import pprint

def card_generator(file_path: str) -> Generator[Dict, None, None]:
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('{') and line.endswith('},'):
                # Remove the trailing comma
                line = line[:-1]
            if line.startswith('{') and line.endswith('}'):
                try:
                    card = json.loads(line)
                    yield card
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")
                    continue

def process_cards(file_path: str) -> List[Dict]:
    processed_cards = []
    for i, card in enumerate(card_generator(file_path), 1):
        processed_card = {
            # 'id': card.get('id'),
            'oracle_id': card.get('oracle_id'),
            # 'multiverse_ids': card.get('multiverse_ids'),
            # 'mtgo_id': card.get('mtgo_id'),
            # 'mtgo_foil_id': card.get('mtgo_foil_id'),
            # 'tcgplayer_id': card.get('tcgplayer_id'),
            # 'cardmarket_id': card.get('cardmarket_id'),
            'name': card.get('name'),
            # 'lang': card.get('lang'),
            'released_at': card.get('released_at'),
            # 'uri': card.get('uri'),
            # 'scryfall_uri': card.get('scryfall_uri'),
            # 'layout': card.get('layout'),
            # 'highres_image': card.get('highres_image'),
            # 'image_status': card.get('image_status'),
            # 'image_uris': card.get('image_uris'),
            'mana_cost': card.get('mana_cost'),
            'cmc': card.get('cmc'),
            'type_line': card.get('type_line'),
            'oracle_text': card.get('oracle_text'),
            'power': card.get('power'),
            'toughness': card.get('toughness'),
            'colors': card.get('colors'),
            'color_identity': card.get('color_identity'),
            'keywords': card.get('keywords'),
            'legalities': card.get('legalities'),
            'games': card.get('games'),
            # 'reserved': card.get('reserved'),
            # 'foil': card.get('foil'),
            # 'nonfoil': card.get('nonfoil'),
            # 'finishes': card.get('finishes'),
            # 'oversized': card.get('oversized'),
            # 'promo': card.get('promo'),
            # 'reprint': card.get('reprint'),
            # 'variation': card.get('variation'),
            # 'set_id': card.get('set_id'),
            # 'set': card.get('set'),
            'set_name': card.get('set_name'),
            # 'set_type': card.get('set_type'),
            # 'set_uri': card.get('set_uri'),
            # 'set_search_uri': card.get('set_search_uri'),
            # 'scryfall_set_uri': card.get('scryfall_set_uri'),
            # 'rulings_uri': card.get('rulings_uri'),
            # 'prints_search_uri': card.get('prints_search_uri'),
            # 'collector_number': card.get('collector_number'),
            # 'digital': card.get('digital'),
            # 'rarity': card.get('rarity'),
            # 'flavor_text': card.get('flavor_text'),
            # 'card_back_id': card.get('card_back_id'),
            # 'artist': card.get('artist'),
            # 'artist_ids': card.get('artist_ids'),
            # 'illustration_id': card.get('illustration_id'),
            # 'border_color': card.get('border_color'),
            # 'frame': card.get('frame'),
            # 'full_art': card.get('full_art'),
            # 'textless': card.get('textless'),
            # 'booster': card.get('booster'),
            # 'story_spotlight': card.get('story_spotlight'),
            # 'edhrec_rank': card.get('edhrec_rank'),
            # 'prices': card.get('prices'),
            # 'related_uris': card.get('related_uris'),
            # 'purchase_uris': card.get('purchase_uris'),
        }
        processed_cards.append(processed_card)
        
        if i % 1000 == 0:
            print(f"Processed {i} cards")
    
    return processed_cards

# Usage
file_path = './data/oracle-cards-20241105220317.json'
cards = process_cards(file_path)

# Print the first few processed cards
pprint(cards[:5])
