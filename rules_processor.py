import re
from typing import List, Dict

def process_rules(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to match rule numbers and their content
    rule_pattern = re.compile(r'(\d+(?:\.\d+)*[a-z]?)\.\s+(.*?)(?=\n\d+(?:\.\d+)*[a-z]?\.\s+|\Z)', re.DOTALL)

    rules = rule_pattern.findall(content)

    processed_rules = []
    current_base_rule = ""
    current_sub_rule = ""

    for rule_number, rule_content in rules:
        parts = rule_number.split('.')
        
        if len(parts) == 1:
            current_base_rule = rule_number
            current_sub_rule = ""
        elif len(parts) == 2:
            current_sub_rule = rule_number
        
        # Check for lettered sub-rules
        sub_rules = re.findall(r'(\d+\.\d+[a-z])\s+(.*?)(?=\n\d+\.\d+[a-z]|\Z)', rule_content, re.DOTALL)
        
        if sub_rules:
            # Process each lettered sub-rule
            for sub_rule_number, sub_rule_content in sub_rules:
                processed_rules.append({
                    'id': f"rule_{sub_rule_number.replace('.', '_')}",
                    'rule_number': sub_rule_number,
                    'content': sub_rule_content.strip(),
                    'base_rule': current_base_rule,
                    'parent_rule': current_sub_rule
                })
        else:
            # Process the rule as a whole
            processed_rules.append({
                'id': f"rule_{rule_number.replace('.', '_')}",
                'rule_number': rule_number,
                'content': rule_content.strip(),
                'base_rule': current_base_rule,
                'parent_rule': current_sub_rule if current_sub_rule else None
            })

    return processed_rules