from typing import List, Optional, TypedDict

class GraphState(TypedDict):
    question: str
    card_names: Optional[List[str]]
    rules: Optional[str]
    game_state: Optional[str]
    response: Optional[str]
