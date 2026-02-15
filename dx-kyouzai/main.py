from fastapi import FastAPI
from dotenv import load_dotenv
import vertexai
from .routers import agent_router

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize Vertex AI
vertexai.init(project="jp-digital-478220", location="asia-northeast1")

app.include_router(agent_router)