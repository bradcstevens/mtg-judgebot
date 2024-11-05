from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from typing import List
import json
from ..api.mtg_cards_api import fetch_card_by_name
from ..api.rules_api import get_rule_and_children
import os

class CardNameRecognitionInput(BaseModel):
    card_names: List[str] = Field(..., description="List of Magic: The Gathering card names to analyze")

class RulesLookupInput(BaseModel):
    rule_numbers: List[str] = Field(..., description="List of Magic: The Gathering rule numbers to look up")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DB_PATH = os.path.join(PROJECT_ROOT, "db", "mtg_cards.sqlite")

# Add this to create db directory if it doesn't exist
os.makedirs(os.path.join(PROJECT_ROOT, "db"), exist_ok=True)

import os

# Update the path construction
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DB_PATH = os.path.join("/deps/__outer_my_agent", "db", "mtg_cards.sqlite")

# Debug prints
print(f"CWD: {os.getcwd()}")
print(f"Script location: {os.path.dirname(__file__)}")
print(f"Project root: {PROJECT_ROOT}")
print(f"DB path: {DB_PATH}")

import logging

logger = logging.getLogger(__name__)

def create_card_name_recognition_tool():
    def recognize_card_names(card_names, db_path="/deps/__outer_my_agent/my_agent/db/mtg_cards.sqlite"):
        logger.info(f"Attempting to recognize cards: {card_names}")
        logger.info(f"Using database path: {db_path}")
        
        try:
            recognized_cards = []
            for card_name in card_names:
                logger.info(f"Looking up card: {card_name}")
                card_details = fetch_card_by_name(db_path, card_name)
                if card_details:
                    recognized_cards.extend(card_details)
                    logger.info(f"Found card: {card_name}")
                else:
                    logger.warning(f"Card not found: {card_name}")
            
            logger.info(f"Recognized cards: {[card['name'] for card in recognized_cards]}")
            return json.dumps(recognized_cards, indent=2)
        except Exception as e:
            logger.error(f"Error in recognize_card_names: {str(e)}")
            raise

    return StructuredTool.from_function(
        func=recognize_card_names,
        name="recognize_card_names",
        description="Pass this tool a list of Magic: The Gathering card names and it will return a list of card details including full text and rulings for the card's abilities. You should call this tool for each thing in user query that sounds like it could be a Magic: The Gathering card name or is being used in the query like a card name would be.",
        args_schema=CardNameRecognitionInput
    )

def create_rules_lookup_tool():
    def rules_lookup(rule_numbers: List[str]) -> str:
        responses = []
        for rule_number in rule_numbers:
            rule_data = get_rule_and_children(rule_number)
            
            if not rule_data:
                responses.append(f"Rule {rule_number} not found.")
            else:
                response = f"Rule {rule_data['rule_number']}: {rule_data['content']}\n\n"
                
                if rule_data['children']:
                    response += "Sub-rules:\n"
                    for child in rule_data['children']:
                        response += f"- {child['rule_number']}: {child['content']}\n"
                else:
                    response += "No sub-rules found.\n"
                
                responses.append(response)
        
        full_response = "\n\n".join(responses)
        print(f"Rules lookup full response:\n{full_response}")
        return full_response

    return StructuredTool.from_function(
        func=rules_lookup,
        name="rules_lookup",
        description="Look up multiple Magic: The Gathering rules by their numbers",
        args_schema=RulesLookupInput
    )

class GameStateConstructor:
    def __init__(self):
        self.description = "Construct a detailed game state based on the user's description"

    def run(self, question: str) -> str:
        # Placeholder for game state construction
        return f"Constructed game state based on: {question}"

# game_state_constructor_tool = Tool(
#     name="game_state_constructor",
#     func=GameStateConstructor().run,
#     description=GameStateConstructor().description
# )
