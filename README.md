# MediBot — Simple Overview

MediBot is a Retrieval‑Augmented Generation (RAG) medical assistant. It answers symptom and diagnosis questions by searching a FAISS vector store of curated medical data and using an LLM agent to reason and synthesize answers.

What this file contains
- Quick EDA highlights used to guide behavior
- Simple architecture description and a compact diagram
- Short inference flow (how a query is handled)
- Minimal notes for a FastAPI + index.html frontend
- Quick local setup and brief AWS hint
- Safety and prompt notes

1 — Exploratory Data Analysis (key points)
- Balanced dataset: 41 diseases, each with 120 records.
- Common symptoms (e.g., fatigue, vomiting, high_fever, nausea, loss_of_appetite) appear across many diseases. When these show up, expect multiple possible diagnoses — list possibilities, don't assert one.
- Severity scores cluster around 3–5 (moderate). Use severity metadata to add nuance (e.g., "moderate severity").

2 — Architecture
- UI: Gradio (app.py) or optional FastAPI + static index.html.
- Orchestration: LangChain + a ReAct-style agent (app.py) that chooses tools and uses conversation memory.
- Tools: tools.py — functions (DiagnosisTool, rag_query) that query FAISS with metadata filters.
- Vector DB: FAISS (faiss_index/) with document metadata like {"type": "diagnosis"}.
- LLM: OpenAI (example: gpt-4o-mini) — used only to reason over retrieved documents and synthesize responses.

Compact diagram (flow)
User (browser / client)
  ↓ HTTP / Gradio
FastAPI or Gradio app (app.py)
  ↓ LangChain ReAct agent
tools.py → rag_query (filter by metadata)
  ↓ FAISS vector store (retrieved docs)
LLM (synthesize) → response to user

3 — Inference flow
- User submits message + session_id.
- Conversation history is attached.
- Agent decides whether to call a tool (e.g., DiagnosisTool) based on the prompt rules (do not use internal knowledge).
- DiagnosisTool calls rag_query(data_type="diagnosis"), which filters FAISS by metadata {"type":"diagnosis"} and returns the most relevant document chunks.
- Agent creates the final reply strictly from retrieved documents. If several diseases match common symptoms, list them as candidates and indicate uncertainty.

4 — FastAPI + index.html
- Suggested endpoints:
  - POST /api/chat — { session_id, message } → { response }
  - GET /api/health — 200 OK
- index.html: a single-page client that posts to /api/chat and appends responses to a chat area. Never store keys in frontend.
- Example fetch (pseudo):
  fetch('/api/chat', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ session_id, message }) })

5 — Quick local setup
- Requirements: Python 3.9+
- Commands:
  - python -m venv .venv
  - Windows: .\.venv\Scripts\activate
  - Mac/Linux: source .venv/bin/activate
  - Create a .env with required keys (OPENAI_API_KEY, optional LANGCHAIN_API_KEY, etc.)
  - pip install -r requirements.txt
  - Run:
    - For Gradio UI: python app.py
    - For FastAPI: uvicorn app:app --host 0.0.0.0 --port 8000
- Open the URL printed by Gradio or use your frontend against the FastAPI endpoint.

6 — Brief AWS hint
- Launch an EC2 (Deep Learning AMI if using GPU), upload the project, create a venv, install requirements, export keys, and run the app. Ensure the security group allows the port (e.g., 7860 or 8000).

7 — Prompting & safety notes
- The system prompt must instruct the agent to rely on tools/FAISS and avoid using or inventing internal knowledge.
- For common symptoms, agent should list candidate diagnoses and show uncertainty.
- Keep secrets in .env and never embed them in static frontend files.
- Review outputs before using for medical decisions; this assistant is informational, not a substitute for professional care.

Files of interest
- app.py — UI + orchestration (Gradio or FastAPI)
- tools.py — DiagnosisTool and rag_query
- loaders.py — CSV → Document objects for FAISS
- faiss_index/ — vector store (if persisted)
- docs/images/ — disease_frequency.png, severity_distribution.png

Contact / next steps
- To change behavior, update the system prompt in app.py and/or refine rag_query filters in tools.py.
- For additional documentation, the code files include comments and examples.
