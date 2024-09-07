import sqlite3
import re

def create_glossary_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS glossary (
        id INTEGER PRIMARY KEY,
        keyword TEXT UNIQUE NOT NULL,
        definition TEXT NOT NULL
    )
    ''')
    conn.commit()

def insert_or_update_glossary_term(conn, keyword, definition):
    cursor = conn.cursor()
    cursor.execute('''
    INSERT OR REPLACE INTO glossary (keyword, definition)
    VALUES (?, ?)
    ''', (keyword, definition))
    conn.commit()

def process_glossary_file(file_path, conn):
    with open(file_path, 'r') as file:
        content = file.read()

    terms = re.split(r'\n\n(?=\S)', content)
    for term in terms:
        lines = term.split('\n')
        keyword = lines[0].strip()
        definition = ' '.join(lines[1:]).strip()
        insert_or_update_glossary_term(conn, keyword, definition)

def create_glossary_db(glossary_file_path, db_path):
    conn = sqlite3.connect(db_path)
    create_glossary_table(conn)
    process_glossary_file(glossary_file_path, conn)
    conn.close()

if __name__ == "__main__":
    glossary_file_path = 'data/glossary.txt'
    db_path = 'db/mtg_rules.sqlite'
    create_glossary_db(glossary_file_path, db_path)
    print("Glossary database created or updated successfully.")