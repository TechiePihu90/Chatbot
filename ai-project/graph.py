"""
graph.py
Layer 4 — the LangGraph state machine that ties RAG (rag.py), tools (tools.py),
and plain chat (backend.py) together into ONE pipeline:

  classify -> (rag | tool | chat) -> [rag can fall back to tool if low-confidence] -> END

Key ideas this layer teaches:
  - Conditional routing: one node decides which path to take next.
  - A feedback loop: RAG can hand off to the tool-agent if it doesn't know the answer.
  - Persistent state: LangGraph's checkpointer remembers the whole conversation
    per `thread_id`, so app.py no longer manages a message list by hand.

This is the ONLY entry point app.py talks to now. app.py doesn't decide
"use RAG" or "use agent" anymore — the graph decides, based on the question.
"""

from typing import TypedDict, Annotated, Optional, List, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from backend import get_llm, generate_response, generate_agent_response
from rag import retrieve_context
from tools import get_tools


# ---- Graph state ----
# `messages` uses the add_messages reducer: every node that returns
# {"messages": [...]} gets those messages APPENDED (not overwritten) to
# history, and the checkpointer persists this automatically per thread_id.
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    route: Optional[str]                 # "rag" | "tool" | "chat" — which path this turn took
    source_docs: Optional[List[Any]]     # populated only if route == "rag"
    tool_steps: Optional[List[Any]]      # populated only if route == "tool"
    needs_fallback: bool                 # True if RAG couldn't answer -> hand off to tools


CLASSIFY_PROMPT = """Classify the user's LATEST question into exactly one category. Reply with ONLY one word, nothing else.

Categories:
- rag: the question is likely about the uploaded document{doc_hint}
- tool: the question needs a calculation, or current/live/real-world info (news, prices, facts you might not reliably know)
- chat: general conversation, greetings, opinions, or something answerable from general knowledge alone

Question: {question}

Answer with exactly one word: rag, tool, or chat."""


def classify_query(state: GraphState, retriever_available: bool) -> dict:
    """
    Entry node. Decides which path this turn takes, and resets the
    per-turn fields (source_docs, tool_steps, needs_fallback) so old
    data from a previous turn doesn't leak into this turn's UI display.
    """
    question = state["messages"][-1].content
    doc_hint = "" if retriever_available else " (NOT available — no document uploaded — never pick rag)"

    llm = get_llm(temperature=0)  # temperature 0 -> consistent, deterministic classification
    prompt = CLASSIFY_PROMPT.format(doc_hint=doc_hint, question=question)
    result = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()

    if "rag" in result and retriever_available:
        route = "rag"
    elif "tool" in result:
        route = "tool"
    else:
        route = "chat"

    return {
        "route": route,
        "source_docs": None,
        "tool_steps": None,
        "needs_fallback": False,
    }


def rag_node(state: GraphState, retriever) -> dict:
    question = state["messages"][-1].content
    context, source_docs = retrieve_context(retriever, question)

    response = generate_response(state["messages"], context=context)

    # Confidence check: RAG_SYSTEM_PROMPT (in backend.py) instructs the model
    # to say this exact phrase when the document doesn't answer the question.
    # We use that as an explicit, checkable confidence signal.
    needs_fallback = "don't have enough information" in response.lower()

    return {
        "messages": [AIMessage(content=response)],
        "source_docs": source_docs,
        "needs_fallback": needs_fallback,
    }


def tool_node(state: GraphState, tools) -> dict:
    messages = state["messages"]

    # If we arrived here as a FALLBACK from rag_node, the low-confidence
    # RAG answer is already the last message — drop it so it doesn't
    # confuse the agent or pollute permanent memory with a wrong answer.
    if state.get("needs_fallback") and isinstance(messages[-1], AIMessage):
        messages = messages[:-1]

    response, steps = generate_agent_response(messages, tools=tools)

    return {
        "messages": [AIMessage(content=response)],
        "tool_steps": steps,
        "needs_fallback": False,
    }


def chat_node(state: GraphState) -> dict:
    response = generate_response(state["messages"])
    return {"messages": [AIMessage(content=response)]}


# ---- Routing functions (used by conditional edges) ----
def route_after_classify(state: GraphState) -> str:
    return state["route"]


def route_after_rag(state: GraphState) -> str:
    return "tool" if state.get("needs_fallback") else END


def build_graph(retriever=None):
    """
    Builds and compiles the graph for the current session.
    Rebuild this whenever `retriever` changes (e.g. a new PDF is processed) —
    the rag/tool nodes close over `retriever`/`tools` at build time.
    """
    tools = get_tools(retriever=retriever)
    retriever_available = retriever is not None

    graph = StateGraph(GraphState)

    graph.add_node("classify", lambda state: classify_query(state, retriever_available))
    graph.add_node("rag", lambda state: rag_node(state, retriever))
    graph.add_node("tool", lambda state: tool_node(state, tools))
    graph.add_node("chat", chat_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"rag": "rag", "tool": "tool", "chat": "chat"},
    )
    graph.add_conditional_edges(
        "rag",
        route_after_rag,
        {"tool": "tool", END: END},
    )
    graph.add_edge("tool", END)
    graph.add_edge("chat", END)

    # MemorySaver = in-memory checkpointer (resets when the app restarts).
    # This is what gives us persistent conversation state per thread_id
    # WITHOUT app.py managing a message list manually.
    # Future upgrade for production: swap for SqliteSaver / PostgresSaver
    # so memory survives app restarts too.
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def seed_messages(user_input: str, is_first_turn: bool) -> list:
    """
    Builds the message list to pass into graph.invoke() for this turn.
    Only the FIRST turn of a thread needs a SystemMessage — after that,
    the checkpointer already has it in persisted history, so we just
    append the new HumanMessage.
    """
    if is_first_turn:
        return [SystemMessage(content="You are a helpful assistant."), HumanMessage(content=user_input)]
    return [HumanMessage(content=user_input)]