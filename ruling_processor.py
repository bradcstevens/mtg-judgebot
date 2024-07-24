from typing import Dict, List
from json_loader import json_loader

def metadata_func(ruling: dict) -> dict:
    return {
            'object': ruling.get('object'),
            'oracle_id': ruling.get('oracle_id'),
            'source': ruling.get('source'),
            'published_at': ruling.get('published_at'),
            'comment': ruling.get('comment'),
    }

def process_rulings(file_path: str) -> List[Dict]:
    return json_loader(
        file_path=file_path,
        content_key='comment',
        metadata_func=metadata_func
    )


