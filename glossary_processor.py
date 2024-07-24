import re
from typing import List, Dict

def process_glossary(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into individual entries
    entries = content.split('\n\n')

    processed_terms = []

    for index, entry in enumerate(entries, start=1):
        lines = entry.split('\n')
        term_name = lines[0].strip()
        definition = ' '.join(lines[1:]).strip()

        # Extract rule references
        rule_refs = re.findall(r'See rule (\d+(?:\.\d+)*[a-z]?)', definition)

        processed_terms.append({
            'id': f"{index}",
            'term': term_name,
            'definition': definition,
            'rule_refs': rule_refs
        })

    return processed_terms