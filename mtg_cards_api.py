import sqlite3
import json
from typing import Dict, Any

def setup_card_database(database_path: str):
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    
    # Create cards table
    c.execute('''CREATE TABLE IF NOT EXISTS cards
                 (oracle_id TEXT PRIMARY KEY, name TEXT, object TEXT, id TEXT,
                 multiverse_ids TEXT, tcgplayer_id INTEGER, cardmarket_id INTEGER,
                 lang TEXT, released_at TEXT, uri TEXT, scryfall_uri TEXT,
                 layout TEXT, highres_image INTEGER, image_status TEXT,
                 image_uris TEXT, mana_cost TEXT, cmc REAL, type_line TEXT,
                 oracle_text TEXT, power TEXT, toughness TEXT, colors TEXT,
                 color_identity TEXT, keywords TEXT, legalities TEXT, games TEXT,
                 reserved INTEGER, foil INTEGER, nonfoil INTEGER, finishes TEXT,
                 oversized INTEGER, promo INTEGER, reprint INTEGER, variation INTEGER,
                 set_id TEXT, set_code TEXT, set_name TEXT, set_type TEXT,
                 set_uri TEXT, set_search_uri TEXT, scryfall_set_uri TEXT,
                 rulings_uri TEXT, prints_search_uri TEXT, collector_number TEXT,
                 digital INTEGER, rarity TEXT, card_back_id TEXT, artist TEXT,
                 artist_ids TEXT, illustration_id TEXT, border_color TEXT,
                 frame TEXT, full_art INTEGER, textless INTEGER, booster INTEGER,
                 story_spotlight INTEGER, edhrec_rank INTEGER, prices TEXT,
                 related_uris TEXT, purchase_uris TEXT)''')

    # Create rulings table
    c.execute('''CREATE TABLE IF NOT EXISTS rulings
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 oracle_id TEXT, object TEXT, source TEXT,
                 published_at TEXT, comment TEXT,
                 FOREIGN KEY (oracle_id) REFERENCES cards(oracle_id))''')

    conn.commit()
    conn.close()

def insert_card_into_db(database_path: str, card: Dict[str, Any]):
    conn = sqlite3.connect(database_path)
    c = conn.cursor()

    # Convert list and dict fields to JSON strings
    for key, value in card.items():
        if isinstance(value, (list, dict)):
            card[key] = json.dumps(value)

    # Create a dictionary with all possible fields, defaulting to None
    default_card = {
        'oracle_id': None, 'name': None, 'object': None, 'id': None,
        'multiverse_ids': None, 'tcgplayer_id': None, 'cardmarket_id': None,
        'lang': None, 'released_at': None, 'uri': None, 'scryfall_uri': None,
        'layout': None, 'highres_image': None, 'image_status': None,
        'image_uris': None, 'mana_cost': None, 'cmc': None, 'type_line': None,
        'oracle_text': None, 'power': None, 'toughness': None, 'colors': None,
        'color_identity': None, 'keywords': None, 'legalities': None, 'games': None,
        'reserved': None, 'foil': None, 'nonfoil': None, 'finishes': None,
        'oversized': None, 'promo': None, 'reprint': None, 'variation': None,
        'set_id': None, 'set': None, 'set_name': None, 'set_type': None,
        'set_uri': None, 'set_search_uri': None, 'scryfall_set_uri': None,
        'rulings_uri': None, 'prints_search_uri': None, 'collector_number': None,
        'digital': None, 'rarity': None, 'card_back_id': None, 'artist': None,
        'artist_ids': None, 'illustration_id': None, 'border_color': None,
        'frame': None, 'full_art': None, 'textless': None, 'booster': None,
        'story_spotlight': None, 'edhrec_rank': None, 'prices': None,
        'related_uris': None, 'purchase_uris': None
    }

    # Update default_card with the actual card data
    default_card.update(card)

    c.execute('''INSERT OR REPLACE INTO cards VALUES (
        :oracle_id, :name, :object, :id, :multiverse_ids, :tcgplayer_id, :cardmarket_id,
        :lang, :released_at, :uri, :scryfall_uri, :layout, :highres_image, :image_status,
        :image_uris, :mana_cost, :cmc, :type_line, :oracle_text, :power, :toughness,
        :colors, :color_identity, :keywords, :legalities, :games, :reserved, :foil,
        :nonfoil, :finishes, :oversized, :promo, :reprint, :variation, :set_id, :set,
        :set_name, :set_type, :set_uri, :set_search_uri, :scryfall_set_uri, :rulings_uri,
        :prints_search_uri, :collector_number, :digital, :rarity, :card_back_id, :artist,
        :artist_ids, :illustration_id, :border_color, :frame, :full_art, :textless,
        :booster, :story_spotlight, :edhrec_rank, :prices, :related_uris, :purchase_uris
    )''', default_card)

    conn.commit()
    conn.close()

def insert_ruling_into_db(database_path: str, ruling: Dict[str, Any]):
    conn = sqlite3.connect(database_path)
    c = conn.cursor()

    c.execute('''INSERT INTO rulings (oracle_id, object, source, published_at, comment)
                 VALUES (:oracle_id, :object, :source, :published_at, :comment)''', ruling)

    conn.commit()
    conn.close()

def fetch_card_details_by_oracle_id(database_path: str, oracle_id: str) -> Dict[str, Any]:
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name
    c = conn.cursor()

    # Fetch card details
    c.execute("SELECT * FROM cards WHERE oracle_id = ?", (oracle_id,))
    card_result = c.fetchone()

    # Fetch associated rulings
    c.execute("SELECT * FROM rulings WHERE oracle_id = ?", (oracle_id,))
    ruling_results = c.fetchall()

    conn.close()

    if card_result:
        card_dict = dict(card_result)

        # Parse JSON strings back to Python objects
        for key, value in card_dict.items():
            if value and isinstance(value, str):
                try:
                    card_dict[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Keep the original string if it's not valid JSON

        # Add rulings to the card dictionary
        card_dict['rulings'] = [
            {
                'object': ruling['object'],
                'source': ruling['source'],
                'published_at': ruling['published_at'],
                'comment': ruling['comment']
            }
            for ruling in ruling_results
        ]

        return card_dict
    else:
        return {}