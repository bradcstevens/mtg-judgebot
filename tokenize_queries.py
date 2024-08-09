import pandas as pd
import json
import uuid
import re

def normalize_apostrophes(text):
    # Replace right single quotation mark (U+2019) with apostrophe
    return text.replace('\u2019', "'")

def parse_csv_to_json(csv_file_path, output_json_path):
    try:
        df = pd.read_csv(csv_file_path)
        result = []
        
        for _, row in df.iterrows():
            body = normalize_apostrophes(row['body'])
            sentences = re.split(r'(?<=[.!?])\s+', body)
            
            for sentence in sentences:
                if '[[' in sentence and ']]' in sentence:
                    tokens = []
                    labels = []
                    parts = re.split(r'(\[\[[^\]]+\]\])', sentence)
                    
                    for part in parts:
                        if part.startswith('[[') and part.endswith(']]'):
                            card_name = part[2:-2]
                            card_tokens = re.findall(r"[\w']+'s|\w+|[.,!?;]", card_name)
                            
                            for i, token in enumerate(card_tokens):
                                tokens.append(token)
                                labels.append('B-CARD' if i == 0 else 'I-CARD')
                        else:
                            other_tokens = re.findall(r"\w+|[.,!?;]", part)
                            tokens.extend(other_tokens)
                            labels.extend(['O'] * len(other_tokens))
                    
                    result.append({
                        'id': row['id'],
                        'chunk_id': str(uuid.uuid4()),
                        'question': sentence,
                        'tokens': tokens,
                        'labels': labels
                    })
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"JSON data saved to {output_json_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Usage
csv_file_path = 'data/mtg_rules_questions.csv'
output_json_path = 'data/prepared_mtg_rules_questions.json'
parse_csv_to_json(csv_file_path, output_json_path)