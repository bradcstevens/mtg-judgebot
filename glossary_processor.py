import re
from typing import List, Dict

def process_glossary(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into individual terms
    terms = re.split(r'\n(?=\S)', content)

    processed_terms = []

    for term in terms:
        # Split the term into name and definition
        parts = term.split('\n', 1)
        if len(parts) == 2:
            term_name, definition = parts
            processed_terms.append({
                'id': f"term_{term_name.lower().replace(' ', '_')}",
                'term': term_name.strip(),
                'definition': definition.strip()
            })

    return processed_terms