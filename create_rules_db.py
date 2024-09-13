import sqlite3
from rules_processor import process_rules
import os

def delete_rules_table(conn):
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS rules')
    conn.commit()

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
    # Delete the existing database file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing database: {db_path}")

    conn = sqlite3.connect(db_path)
    
    # Delete the existing rules table if it exists
    delete_rules_table(conn)
    print("Deleted existing rules table")
    
    # Create a new rules table
    create_rules_table(conn)
    print("Created new rules table")
    
    rules = process_rules(rules_file_path)
    for rule in rules:
        insert_or_update_rule(conn, rule['rule_number'], rule['content'], rule['parent_rule'])
    
    conn.close()
    print("Rules database created and populated successfully.")

if __name__ == "__main__":
    rules_file_path = 'data/official-rules.txt'
    db_path = 'db/mtg_rules.sqlite'
    create_rules_db(rules_file_path, db_path)