import sqlite3

def get_glossary_term(keyword):
    conn = sqlite3.connect('db/mtg_rules.sqlite')
    cursor = conn.cursor()
    
    cursor.execute('SELECT definition FROM glossary WHERE keyword = ?', (keyword,))
    result = cursor.fetchone()
    
    conn.close()
    
    return result[0] if result else None