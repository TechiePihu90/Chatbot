"""
tools.py
Defines the tools an agent can call: web search, calculator, and (optionally)
document search over whatever PDF was loaded via rag.py.

Kept separate from backend.py so tool definitions can grow/change independently
of the agent-running logic in backend.py.
"""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


@tool
def web_search(query: str) -> str:
    """Search the web for current information not available in local documents.
    Use this for general knowledge questions, current events, or anything
    outside the user's uploaded documents."""
    search = DuckDuckGoSearchRun()
    return search.run(query)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '23 * 47' or '(100 - 15) / 5'.
    Only use this for arithmetic, not for general questions."""
    try:
        # Restricted eval: only allow digits, operators, parentheses, decimal points.
        # This is NOT bulletproof against all malicious input, but is fine for a
        # learning project where the "attacker" is just you testing edge cases.
        allowed_chars = set("0123456789+-*/(). ")
        if not set(expression) <= allowed_chars:
            return "Error: expression contains disallowed characters."
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def build_doc_search_tool(retriever):
    """
    Wraps the Layer 2 RAG retriever as an agent tool. This is the key idea
    of this layer: instead of WE deciding "use RAG or not" (like in app.py
    during Layer 2), the AGENT now decides for itself, based on the tool's
    description, whether the question needs the document.
    """
    from rag import retrieve_context

    @tool
    def doc_search(query: str) -> str:
        """Search the user's uploaded document for relevant information.
        Use this FIRST for any question that might relate to the uploaded PDF,
        before falling back to web_search."""
        context, _ = retrieve_context(retriever, query)
        return context if context else "No relevant information found in the document."

    return doc_search


def get_tools(retriever=None):
    """
    Returns the list of tools available to the agent.
    doc_search is only included if a retriever (i.e. a processed PDF) exists —
    this is how Layer 2 (RAG) and Layer 3 (agent) plug into each other.
    """
    tools = [web_search, calculator]
    if retriever:
        tools.append(build_doc_search_tool(retriever))
    return tools