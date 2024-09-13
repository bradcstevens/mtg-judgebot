import re
from typing import List, Dict

def process_rules(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to match rule numbers and their content
    rule_pattern = re.compile(r'(\d+(?:\.\d+)*[a-z]?)\.\s+(.*?)(?=\n\d+(?:\.\d+)*[a-z]?\.\s+|\Z)', re.DOTALL)

    rules = rule_pattern.findall(content)

    processed_rules = []

    for rule_number, rule_content in rules:
        parts = rule_number.split('.')
        
        if len(parts) == 1:
            parent_rule = None
        else:
            parent_rule = '.'.join(parts[:-1])
        
        processed_rules.append({
            'id': f"rule_{rule_number.replace('.', '_')}",
            'rule_number': rule_number,
            'content': rule_content.strip(),
            'parent_rule': parent_rule
        })

    return processed_rules