from langchain_openai import ChatOpenAI  # Changed from ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from .state import GraphState
from .tools import create_card_name_recognition_tool, create_rules_lookup_tool
import json
from typing import Union, Sequence, Annotated
from langgraph.prebuilt import ToolExecutor
tool_belt = [
    create_card_name_recognition_tool(),
]

tool_executor = ToolExecutor(tool_belt)

from langchain_core.utils.function_calling import convert_to_openai_function

functions = [convert_to_openai_function(t) for t in tool_belt]
model = ChatOpenAI(
    temperature=0,
    model="gpt-4o"  # or gpt-3.5-turbo if preferred
).bind_functions(functions)

def call_model(state):
    messages = state.get("messages", [])
    response = model.invoke(messages)
    return {"messages": messages + [response]}

from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage

def call_tool(state):
  last_message = state["messages"][-1]

  action = ToolInvocation(
      tool=last_message.additional_kwargs["function_call"]["name"],
      tool_input=json.loads(
          last_message.additional_kwargs["function_call"]["arguments"]
      )
  )

  response = tool_executor.invoke(action)

  function_message = FunctionMessage(content=str(response), name=action.tool)

  return {"messages" : [function_message]}

def card_name_recognition(state: GraphState) -> GraphState:
    card_name_tool = create_card_name_recognition_tool()
    result = card_name_tool.run({"card_names": [state["response"]]}, db_path="/deps/__outer_my_agent/my_agent/db/mtg_cards.sqlite")
    state["card_names"] = json.loads(result)
    return state

def rules_lookup_node(state: GraphState) -> GraphState:
    rules_lookup_tool = create_rules_lookup_tool()
    result = rules_lookup_tool.run(["100"])
    state["rules"] = result
    return state

# def game_state_construction(state: GraphState) -> GraphState:
#     game_state_constructor = GameStateConstructor()
#     result = game_state_constructor.run(state["question"])
#     state["game_state"] = result
#     return state

def agent_execution(state: GraphState) -> Union[GraphState, Sequence[Annotated[GraphState, "final_answer"]]]:
    # Implementation of agent execution
    pass