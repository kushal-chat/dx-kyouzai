from fastapi import APIRouter, FastAPI
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition

from vertexai import rag
import vertexai

from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

# Initialize Vertex AI
vertexai.init(project="jp-digital-478220", location="asia-northeast1")

# Initialize FastAPI
app = FastAPI()
router = APIRouter(prefix="/agent")

# Initialize chat model
response_model = init_chat_model("google_vertexai:gemini-2.5-flash", temperature=0)


@tool
def rag_retrieval_tool(query: str):
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
    return response


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = response_model.bind_tools([rag_retrieval_tool]).invoke(state["messages"])
    return {"messages": [response]}


@router.post("/")
async def chat_endpoint(query: str):

    builder = StateGraph(MessagesState)

    builder.add_node(generate_query_or_respond)
    builder.add_node("rag_retrieval", ToolNode(tools=[rag_retrieval_tool]))

    builder.add_edge(START, "generate_query_or_respond")

    # Decide whether to retrieve
    builder.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "rag_retrieval",
            END: END,
        },
    )

    graph = builder.compile()

    for chunk in graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{query}",
                }
            ]
        }
    ):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n\n")


app.include_router(router)
