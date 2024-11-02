import sqlite3
import logging

# Add this at the top of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_rule_and_children(rule_number):
    try:
        conn = sqlite3.connect('db/mtg_rules.sqlite')
        cursor = conn.cursor()
        
        logger.info(f"Fetching rule {rule_number}")
        
        # Get the main rule
        cursor.execute('SELECT rule_number, content FROM rules WHERE rule_number = ?', (rule_number,))
        main_rule = cursor.fetchone()
        
        if not main_rule:
            logger.warning(f"Rule {rule_number} not found in database")
            return None
        
        result = {
            'rule_number': main_rule[0],
            'content': main_rule[1],
            'children': []
        }
        
        logger.info(f"Main rule {rule_number} found: {result['content'][:50]}...")
        
        # Get child rules
        cursor.execute('SELECT rule_number, content FROM rules WHERE parent_rule = ?', (rule_number,))
        children = cursor.fetchall()
        
        for child in children:
            result['children'].append({
                'rule_number': child[0],
                'content': child[1]
            })
        
        logger.info(f"Found {len(result['children'])} child rules for {rule_number}")
        
        return result
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()

# Add a function to check database connection and content
def check_database():
    try:
        conn = sqlite3.connect('db/mtg_rules.sqlite')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM rules')
        count = cursor.fetchone()[0]
        logger.info(f"Total rules in database: {count}")
        
        cursor.execute('SELECT rule_number, content FROM rules LIMIT 5')
        sample = cursor.fetchall()
        logger.info("Sample rules:")
        for rule in sample:
            logger.info(f"{rule[0]}: {rule[1][:50]}...")
        
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return False
    finally:
        if conn:
            conn.close()