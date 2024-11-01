from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from typing import List
import json
from my_agent.utils.api.mtg_cards_api import fetch_card_by_name
from my_agent.utils.api.rules_api import get_rule_and_children

class CardNameRecognitionInput(BaseModel):
    card_names: List[str] = Field(..., description="List of Magic: The Gathering card names to analyze")

class RulesLookupInput(BaseModel):
    rule_numbers: List[str] = Field(..., description="List of Magic: The Gathering rule numbers to look up")

def create_card_name_recognition_tool():
    def recognize_card_names(card_names):
        recognized_cards = []
        for card_name in card_names:
            card_details = fetch_card_by_name("../db/mtg_cards.sqlite", card_name)
            if card_details:
                recognized_cards.extend(card_details)
            else:
                print(f"Card not found: {card_name}")
        
        print(f"Recognized cards: {[card['name'] for card in recognized_cards]}")
        return json.dumps(recognized_cards, indent=2)

    return StructuredTool.from_function(
        func=recognize_card_names,
        name="recognize_card_names",
        description="Recognize and log Magic: The Gathering card names from the user's input.",
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

game_state_constructor_tool = Tool(
    name="game_state_constructor",
    func=GameStateConstructor().run,
    description=GameStateConstructor().description
)
