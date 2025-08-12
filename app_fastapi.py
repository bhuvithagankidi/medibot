# C:/Users/bhuvi/PycharmProjects/medi-bot/app.py

import os
import uuid
import logging
from dotenv import load_dotenv

# --- FastAPI Imports ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- LangChain Imports ---
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- Project-specific Imports ---
from tools import get_all_tools
from loaders import load_medical_docs

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- LangChain & AI Model Initialization ---
# LangChain components automatically find API keys in the environment.
MODEL = "gpt-4o-mini"
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model=MODEL, temperature=0)

# --- FAISS Vector Store Initialization ---
FAISS_INDEX_PATH = "./faiss_index"
try:
    if os.path.exists(FAISS_INDEX_PATH):
        logging.info(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        vector = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        logging.info("FAISS index loaded successfully.")
    else:
        logging.info("No FAISS index found. Building a new one from documents...")
        docs = load_medical_docs()
        if not docs:
            raise RuntimeError("No documents were loaded. Please check your 'data' directory.")
        logging.info(f"Creating FAISS index from {len(docs)} documents...")
        vector = FAISS.from_documents(docs, embeddings)
        vector.save_local(FAISS_INDEX_PATH)
        logging.info(f"New FAISS index created and saved to '{FAISS_INDEX_PATH}'.")
except Exception as e:
    raise RuntimeError(f"Failed to load or create FAISS index. Error: {e}")

# --- Agent and Tools Setup ---
retriever = vector.as_retriever()
tools = get_all_tools(retriever)
prompt = hub.pull("hwchase17/react")

# Define your custom system message
system_message = """You are MediBot, a friendly and highly detailed AI-powered symptom checker..."""  # (Your detailed prompt here)
prompt.template = system_message + "\n\n" + prompt.template

react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)

# --- Session and History Management ---
session_storage = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Fetches or creates a chat history instance for a given session."""
    if session_id not in session_storage:
        session_storage[session_id] = ChatMessageHistory()
    return session_storage[session_id]


agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- FastAPI Application ---
app = FastAPI(
    title="MediBot API",
    description="An API for interacting with the AI-powered medical assistant.",
    version="1.0.0"
)

# --- Add CORS Middleware ---
# This is the new block you need to add.
# It allows your frontend (running on any domain, indicated by "*")
# to communicate with your backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --- Pydantic Models for Request and Response ---
class ChatRequest(BaseModel):
    user_input: str = Field(..., description="The user's message to the chatbot.")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()),
                            description="A unique identifier for the conversation session.")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="The chatbot's response.")
    session_id: str = Field(..., description="The session ID for the conversation, returned for stateful clients.")


# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handles a user's chat message and returns the agent's response.
    """
    logging.info(f"Received request for session_id: {request.session_id}")

    # Invoke the agent with the correct structure for RunnableWithMessageHistory
    response = agent_with_chat_history.invoke(
        {"input": request.user_input},
        config={"configurable": {"session_id": request.session_id}}
    )

    answer = response.get("output", "Sorry, I encountered an error.")

    return ChatResponse(answer=answer, session_id=request.session_id)

# To run the app, use the command: uvicorn app:app --reload