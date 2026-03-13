# KrishiSahayak - AI Agri Scheme Assistant

## Use Case Objective
KrishiSahayak is a Streamlit-based chatbot designed to help Indian farmers understand government agricultural schemes quickly and accurately. The system combines local knowledge base retrieval (RAG) with optional live web search to provide grounded answers with sources.

## How We Approached The Problem
1. Defined a focused domain: Indian agriculture schemes.
2. Built a modular architecture using config, models, and utils.
3. Implemented Retrieval-Augmented Generation (RAG) with ChromaDB for local documents.
4. Added web search fallback (Serper, then DuckDuckGo) for fresh information.
5. Added response modes (Concise and Detailed) for user-friendly control.
6. Added citations in responses for transparency.

## Solution
The app loads and chunks documents from `data/`, builds/loads embeddings into ChromaDB, retrieves relevant context per query, combines it with optional web results, and sends a grounded prompt to the selected LLM provider (Gemini, OpenAI, or Groq).

## Features Implemented
- RAG pipeline with local documents (PDF/TXT/MD/CSV)
- Persistent vector store with ChromaDB
- Embedding model support via `models/embeddings.py`
- Multi-LLM provider support via `models/llm.py`:
  - Gemini default
  - OpenAI
  - Groq
- Live web search integration:
  - Serper API 
  - DuckDuckGo fallback
- Response mode toggle in UI:
  - Concise
  - Detailed
- Source citations shown in chat responses
- Secure environment variable handling for API keys

## Challenges Faced
- PDF parsing warnings in some source files (non-standard PDF structure)
- Embedding quota/rate limit while building large indexes with Gemini free tier
- Need to manage rebuild strategy for deployment environments with ephemeral storage

## Deployment Link
Add your Streamlit Cloud URL here after deployment:

[Open Streamlit APP](https://kisansahayak.streamlit.app/)

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables
Create a `.env` file (or configure Streamlit Secrets) with:
- `GEMINI_API_KEY`
- `OPENAI_API_KEY` (optional)
- `GROQ_API_KEY` (optional)
- `SERPER_API_KEY` (optional)

