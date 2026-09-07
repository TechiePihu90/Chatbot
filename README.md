# Personal Knowledge Assistant

An agentic knowledge assistant built to learn LangChain, LangGraph, and RAG fundamentals from the ground up — not by following one tutorial, but by building a system in layers, where each layer solves a real limitation of the one before it.

Chat normally, upload a PDF and get grounded (non-hallucinated) answers, or ask something needing a calculation or current info — a LangGraph pipeline automatically figures out how to answer, with full conversation memory.

## What it does

- **Chat normally** — general conversation, no lookup needed
- **Chat with your PDF** — upload a document, get answers grounded strictly in its content, with page-level sources shown
- **Use tools automatically** — web search and a calculator, called only when the question needs them
- **Route itself** — a LangGraph state machine classifies each question and sends it down the right path: `rag`, `tool`, or `chat`
- **Fall back when unsure** — if the document doesn't actually answer the question, RAG hands off to the tool-agent instead of guessing or giving up
- **Remember the conversation** — LangGraph's checkpointer persists full message history per session, no manual list-management in the UI

## Architecture

```
                    ┌──────────────┐
                    │  classify    │   (LLM decides: rag / tool / chat)
                    └──────┬───────┘
           ┌───────────────┼───────────────┐
           ▼                ▼                ▼
      ┌─────────┐      ┌─────────┐      ┌─────────┐
      │   rag   │      │  tool   │      │  chat   │
      │ (answer │      │ (agent  │      │(general │
      │ from PDF)│      │w/ tools)│      │  chat)  │
      └────┬────┘      └────┬────┘      └────┬────┘
           │                 ▲                │
   low confidence?           │                │
           └─────fallback────┘                │
           │                                  │
          END ◄──────────────────────────────END
```

Built with **LangGraph** as an explicit state machine rather than a plain chain, so routing is automatic, inspectable, and can loop — e.g. RAG handing off to the tool-agent when the document doesn't have the answer, instead of returning a low-confidence guess.

## Tech stack

| Component          | Choice                                          |
|---------------------|--------------------------------------------------|
| LLM                 | Groq (`openai/gpt-oss-20b`)                        |
| Orchestration       | LangChain + LangGraph                               |
| Embeddings          | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector store        | Chroma (local, persisted to disk)                    |
| Web search tool     | DuckDuckGo (free, no API key)                          |
| UI                  | Streamlit                                                |
| State persistence   | LangGraph `MemorySaver` checkpointer, keyed by `thread_id` |

## Project structure

```
chatbot/
├── app.py            # Streamlit UI — sends messages into the graph, renders results + transparency panel
├── graph.py           # LangGraph state machine: classify → rag/tool/chat → confidence fallback → END
├── backend.py          # LLM setup, system prompts, plain chat + agent-executor logic
├── rag.py               # PDF loading, chunking, embeddings, Chroma vector store, retrieval
├── tools.py               # Agent tools: web_search, calculator, doc_search (wraps the RAG retriever)
├── requirements.txt
└── .env                     # GROQ_API_KEY (not committed)
```

Each file has one job. `backend.py` is the shared toolkit (LLM instance, prompts, agent construction) that `graph.py` and `app.py` both build on — this is what let the project grow from a single-file chatbot into a multi-layer pipeline without a rewrite at every stage.

## How this was built (in layers)

1. **Basic chatbot** — LangChain + Groq + Streamlit, with conversation memory (a running list of messages passed to every LLM call).
2. **RAG** — PDF loading → chunking → embeddings → Chroma vector store → retrieval, with a strict "only answer from context, otherwise say you don't know" system prompt to reduce hallucination.
3. **Agentic tool-calling** — wrapped web search, a calculator, and the RAG retriever as LangChain tools; a tool-calling agent decides for itself which (if any) to use, instead of hardcoded if/else logic.
4. **LangGraph state machine** *(current layer)* — replaced manual routing with an explicit graph: automatic query classification (`classify` node), conditional edges to `rag` / `tool` / `chat`, and a fallback loop where a low-confidence RAG answer routes to the tool-agent instead of being returned as-is. Conversation state is checkpointed per `thread_id`, so `app.py` no longer manages a message list by hand — it just reads the graph's state.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

## Run the app

```bash
streamlit run app.py
```

Optionally upload a PDF in the sidebar and click **Process PDF** — this rebuilds the graph so its `rag`/`tool` nodes pick up the new retriever. Then just chat. Below each answer, an expandable panel shows **how** that turn was answered: source pages (RAG), tool calls made (agent), or a note that it was answered from general knowledge (chat).

## What this project actually demonstrates

- Grounding LLM answers in real documents instead of relying on parametric knowledge (RAG)
- Letting a model decide when to call tools instead of hardcoding branches (agentic tool-calling)
- Explicit, inspectable control flow with conditional routing and a fallback loop (LangGraph over a plain chain)
- Persistent, checkpointed conversation state instead of a hand-rolled message list




