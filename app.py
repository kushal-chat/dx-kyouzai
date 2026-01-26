from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel

from typing import List, Dict, Literal

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
)

class ProjectState(BaseModel):
    school_name: str
    images: List
    school_level: List[Literal['小', '中', '高']]
    prefecture: str
    project_form: str
    details: Dict

builder = StateGraph(ProjectState)

builder.add_node()