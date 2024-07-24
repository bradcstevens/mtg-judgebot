import json
from typing import Dict, List, Callable

def json_loader(file_path: str, 
                content_key: str,
                metadata_func: Callable[[Dict], Dict]) -> List[Dict]:
    processed_data = []
    
    with open(file_path, 'r') as file:
        data = json.load(file)
        for seq_num, obj in enumerate(data, 1):
            content = obj.get(content_key, '')
            metadata = metadata_func(obj)
            metadata['source'] = file_path
            metadata['seq_num'] = seq_num
            
            processed_data.append({
                'content': content,
                'metadata': metadata
            })

    return processed_data