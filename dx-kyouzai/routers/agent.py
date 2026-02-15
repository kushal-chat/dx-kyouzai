from __future__ import annotations

# LangChain + LangGraph
from langchain.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import tools_condition
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

# FastAPI + Typing
from fastapi import APIRouter
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.tools import tool
from vertexai import rag
import os

@tool
def retrieval_tool(query: str):
    """
    Run Vertex AI RAG Engine using this tool.
    """
    rag_retrieval_config = rag.RagRetrievalConfig(
        top_k=5,
        filter=rag.Filter(vector_distance_threshold=0.5),
    )
    response = rag.retrieval_query(
        rag_resources=[
            rag.RagResource(
                rag_corpus=os.environ.get("GCP_RAG_CORPUS"),
            )
        ],
        text=query,
        rag_retrieval_config=rag_retrieval_config,
    )
    return {"projects": [response]}
    
# State
class KyouzaiState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # Message Passing
    

# Router
router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    responses={404: {"description": "Not found"}},
)

# Initialize chat model
response_model = init_chat_model("google_vertexai:gemini-2.5-flash", temperature=0)


def invoke_llm(state: KyouzaiState):
    """
    Call the model to generate a response based on the current state.
    Given the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = response_model.bind_tools([retrieval_tool]).invoke(state["messages"])
    return {"messages": [response]}


@router.post("/")
async def chat_endpoint(query: str):
    """
    Endpoint to chat with LLM and call RAG.
    """
    builder = StateGraph(KyouzaiState)
    builder.add_node("invoke_llm", invoke_llm)
    builder.add_edge(START, "invoke_llm")
    builder.add_conditional_edges(
        tools_condition,

    )
    builder.add_edge("invoke_llm", END)
    graph = builder.compile()
    
    return graph.invoke({"messages": HumanMessage(content=query)})