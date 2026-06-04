# Equipment Q&A Agent — Local PoC
# LangGraph + ChromaDB + GPT-4o + FastAPI + Streamlit

## Project structure

poc/
├── requirements.txt
├── .env.example
└── agent/
    ├── setup_vectorstore.py   # Step 1 — load docs into ChromaDB
    ├── agent.py               # Step 2 — LangGraph agent (THIS FILE)
    ├── api.py                 # Step 3 — FastAPI wrapper
    └── ui.py                  # Step 4 — Streamlit chat UI

## Setup (5 minutes)

# 1. Create virtual environment
python -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
cp .env.example .env
# Edit .env and paste your key from platform.openai.com

# 4. Load sample documents into ChromaDB
python agent/setup_vectorstore.py

# 5. Test the agent directly
python agent/agent.py "What caused the P-101 incidents in January 2024?"

## Try these test queries
# "What are the common failure modes for centrifugal pumps?"
# "What happened to pump P-101 in January 2024?"
# "What vibration level requires immediate shutdown?"
# "How often should I regrease the P-101 bearings?"
# "What is the lubrication spec for pump P-101?"
