from .state import GraphState
from langchain.tools.base import StructuredTool
from .tools import create_card_name_recognition_tool, create_rules_lookup_tool, RulesLookupInput, GameStateConstructor
import json
from typing import Union, Sequence, Annotated

def card_name_recognition(state: GraphState) -> GraphState:
    card_name_tool = create_card_name_recognition_tool()
    result = card_name_tool.run(state["question"])
    state["card_names"] = json.loads(result)
    return state

def rules_lookup_node(state: GraphState) -> GraphState:
    rules_lookup_tool = create_rules_lookup_tool()
    result = rules_lookup_tool.run(["100"])
    state["rules"] = result
    return state

def game_state_construction(state: GraphState) -> GraphState:
    game_state_constructor = GameStateConstructor()
    result = game_state_constructor.run(state["question"])
    state["game_state"] = result
    return state

def agent_execution(state: GraphState) -> Union[GraphState, Sequence[Annotated[GraphState, "final_answer"]]]:
    # Implementation of agent execution
    pass
