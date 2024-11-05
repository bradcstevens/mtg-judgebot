from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage
from my_agent.utils.state import GraphState
from my_agent.utils.nodes import call_model, call_tool, card_name_recognition, rules_lookup_node, agent_execution
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

# Define the config
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str

def create_graph():
    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)
    workflow.set_entry_point("agent")

    # workflow.add_node("card_name_recognition", card_name_recognition)
    # workflow.add_node("rules_lookup", rules_lookup_node)
    # workflow.add_node("game_state_construction", game_state_construction)
    # workflow.add_node("agent_execution", agent_execution)

    def should_continue(state):
        last_message = state["messages"][-1]
        if "function_call" not in last_message.additional_kwargs:
            return "end"
        return "continue"

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END
        }
    )
    workflow.add_edge("action", "agent")
    return workflow.compile()

# Export the graph instance that langgraph will use
graph = create_graph()

