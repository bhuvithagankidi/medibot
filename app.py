
import os
import uuid
import logging

from dotenv import load_dotenv
from langchain import hub  #
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from langchain_core.agents import AgentAction
from langchain_core.documents import Document
import gradio as gr
from tools import get_all_tools
from loaders import load_medical_docs


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Now you can safely access the keys. LangChain components will
# automatically pick up the keys from the environment, so you don't
# need to pass them manually everywhere.
# OPENAI_API_KEY = os.getenv("Open_API_Key")
# TAVILY_API_KEY = os.getenv("Tavily_API_Key")

# --- Load Environment Variables FIRST ---
# This will load the variables from your .env file into the environment
load_dotenv()

## Set the OpenAI API key and model name
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT") # e.g., "MediBot-Prod"
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
MODEL="gpt-4o-mini"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# LangChain components automatically find API keys in the environment.
embeddings = OpenAIEmbeddings()

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
            raise RuntimeError(
                "No documents were loaded. Please check your 'data' directory."
            )
        logging.info(f"Creating FAISS index from {len(docs)} documents...")
        vector = FAISS.from_documents(docs, embeddings)
        vector.save_local(FAISS_INDEX_PATH)
        logging.info(f"New FAISS index created and saved to '{FAISS_INDEX_PATH}'.")
except Exception as e:
    raise RuntimeError(f"Failed to load or create FAISS index. Error: {e}")

## Create the conversational agent

llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], temperature=0)

# Creating a retriever
# See https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/vectorstore/
retriever = vector.as_retriever()

## get list of tools
tools = get_all_tools(retriever)
tool_map = {tool.name: tool for tool in tools}

# hwchase17/react is a prompt template designed for ReAct-style
# conversational agents
prompt = hub.pull("hwchase17/react")
# Modify the system message within the prompt to change the agent's personality

# Define your custom system message
system_message =  """You are MediBot, a friendly and highly detailed AI-powered symptom checker.

Your primary goal is to assist users by providing comprehensive, easy-to-understand, and elaborated answers.

Here are your instructions:
1.  **Analyze the user's input.** If it is a greeting, a thank you, or a simple conversational phrase, DO NOT use a tool.
    Respond politely and conversationally.
2.  **For ANY factual question or query requiring information from your knowledge base, your ABSOLUTE FIRST STEP MUST BE to use the appropriate tool to retrieve relevant information.**
    *   **NEVER** rely on your internal, pre-trained knowledge for factual answers. All factual answers MUST be directly and solely derived from the 'Observation' provided by your tools.
    *   If the user asks a factual question, your 'Thought' should immediately lead to an 'Action' using a retrieval tool.
3.  **If you use a tool,** synthesize the information from the 'Observation' and present it in a helpful, reassuring
    and conversational tone in your 'Final Answer'.
    *   Your 'Final Answer' for factual questions MUST be directly supported by, and strictly limited to, the 'Observation' from your tools. Do not add any information not explicitly present in the observation.
4.  **CRITICAL RULE: If a tool returns an unhelpful response like "I don't know" or no relevant information,**
    you MUST NOT try another tool. Your Final Answer must be: "I'm sorry, but I couldn't find specific information about that in my knowledge base."
    Do not apologize for not having an answer multiple times. Stop processing further.
5.  **If a user's query is too short or unclear** (e.g., just "fever"), ask for more details to provide a better answer.
6.  **Do Not hallucinate.** Provide the answer based *only* on the retrieved documents. If there is no relevant information in the knowledge base, you must inform the user that you couldn't find specific information about their query.
7.  **if multiple diseases for a set of common symptoms, list all diseases as possibilities rather than confidently picking one.
8.  **Always provide a Final Answer** that is clear, concise, and directly addresses the user's question, grounded in the retrieved facts.

You have access to the following tools:"""

# Prepend your custom message to the existing template instructions
prompt.template = system_message + "\n\n" + prompt.template

# Create the ReAct agent
react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Configure the AgentExecutor
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)

# Session and History Management ---

# Use a simple dictionary for in-memory session storage
session_storage = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Fetches or creates a chat history instance for a given session."""
    if session_id not in session_storage:
        session_storage[session_id] = ChatMessageHistory()
    return session_storage[session_id]

# Wrap the agent with session-based chat history management.
# This is the modern, correct way to handle conversation memory.
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history", # This key must match the prompt's variable name
)

def chat_with_agent(user_input: str, chat_history: list, session_id: str) -> str:
    """
    Processes user input. The 'chat_history' parameter is provided by Gradio's
    ChatInterface but is not used here, as LangChain's RunnableWithMessageHistory
    manages the history internally.
    """
    # Invoke the agent with the correct structure for RunnableWithMessageHistory
    response = agent_with_chat_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    # The response from AgentExecutor is a dictionary. We need the 'output' value.
    return response.get("output", "Sorry, I encountered an error.")


if __name__ == "__main__":
    # Create Gradio app interface
    with gr.Blocks() as app:
        gr.Markdown("# 🤖 MediBot - Agents & ReAct Framework")
        gr.Markdown("Enter your query below and get AI-powered responses with session memory.")

        # A hidden State component to store a unique session ID for each user conversation
        session_id_state = gr.State(lambda: str(uuid.uuid4()))

        # Use gr.ChatInterface for a clean, pre-built UI
        gr.ChatInterface(
            fn=chat_with_agent,
            title="MediBot",
            description="Ask me about product reviews!",
            additional_inputs=[session_id_state],  # Pass the session_id to the function
        )
    # Launch the Gradio app
    app.launch(debug=True, share=True)

@traceable(name="evaluation_chain")
def get_response_for_evaluation(user_input: str, session_id: str) -> dict:
    """
    A special function for evaluation that returns the agent's answer
    AND the documents that were *actually* retrieved by the agent for context.
    """
    # Invoke the agent to get the full response, including intermediate steps
    response = agent_with_chat_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    # Extract the final answer
    answer = response.get("output", "")

    # Correctly extract documents from the agent's actual tool calls
    retrieved_docs = []
    if "intermediate_steps" in response:
        for action, observation in response["intermediate_steps"]:
            # The 'observation' is a string. To get the raw documents for evaluation,
            # we must re-run the tool function that the agent chose to call.
            if action.tool in tool_map:
                tool_to_rerun = tool_map[action.tool]
                try:
                    # This call gets the actual List[Document]
                    docs_from_tool = tool_to_rerun.func(action.tool_input)
                    if isinstance(docs_from_tool, list) and all(isinstance(doc, Document) for doc in docs_from_tool):
                        retrieved_docs.extend(docs_from_tool)
                except Exception as e:
                    logging.error(f"Could not re-run tool '{action.tool}' for evaluation. Error: {e}")

    return {
        "answer": answer,
        "documents": retrieved_docs,
    }

