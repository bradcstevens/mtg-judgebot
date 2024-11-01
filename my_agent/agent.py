from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from my_agent.utils.state import GraphState
from my_agent.utils.nodes import card_name_recognition, rules_lookup_node, game_state_construction, agent_execution

# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

workflow = StateGraph(GraphState, config_schema=GraphConfig)
workflow.add_node("card_name_recognition", card_name_recognition)
workflow.add_node("rules_lookup", rules_lookup_node)
workflow.add_node("game_state_construction", game_state_construction)
workflow.add_node("agent_execution", agent_execution)

workflow.set_entry_point("card_name_recognition")
workflow.add_edge("card_name_recognition", "rules_lookup")
workflow.add_edge("rules_lookup", "game_state_construction")
workflow.add_edge("game_state_construction", "agent_execution")
workflow.add_edge("agent_execution", END)

graph = workflow.compile()