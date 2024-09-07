import sqlite3
from rules_processor import process_rules

def create_rules_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS rules (
        id INTEGER PRIMARY KEY,
        rule_number TEXT UNIQUE NOT NULL,
        content TEXT NOT NULL,
        parent_rule TEXT
    )
    ''')
    conn.commit()

def insert_or_update_rule(conn, rule_number, content, parent_rule):
    cursor = conn.cursor()
    cursor.execute('''
    INSERT OR REPLACE INTO rules (rule_number, content, parent_rule)
    VALUES (?, ?, ?)
    ''', (rule_number, content, parent_rule))
    conn.commit()

def create_rules_db(rules_file_path, db_path):
    conn = sqlite3.connect(db_path)
    create_rules_table(conn)
    
    rules = process_rules(rules_file_path)
    for rule in rules:
        insert_or_update_rule(conn, rule['rule_number'], rule['content'], rule['parent_rule'])
    
    conn.close()

if __name__ == "__main__":
    rules_file_path = 'data/official-rules.txt'
    db_path = 'db/mtg_rules.sqlite'
    create_rules_db(rules_file_path, db_path)
    print("Rules database created or updated successfully.")