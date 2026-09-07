"""
backend.py
All chatbot "engine" logic lives here — LLM setup, memory format, response generation.
This file will grow as we add layers (RAG, tools, LangGraph) without touching the UI code.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Modern LangChain 1.x agent syntax
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()

# ---- LLM setup ----

def get_llm(model_name: str = "openai/gpt-oss-20b", temperature: float = 0.7):
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_name,
        temperature=temperature,
    )


DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."

# System prompt used ONLY when RAG context is available.
# The "don't know" instruction is important — this is what stops
# hallucination when the retrieved chunks don't actually answer the question.
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context below.

Rules:
- If the answer is present in the context, answer clearly and cite which part of the context you used.
- If the context does NOT contain enough information to answer, say "I don't have enough information in the provided documents to answer that." Do NOT make up an answer.
- Keep answers concise.

Context:
{context}
"""


def get_initial_messages():
    """Returns the starting message list (just the system prompt)."""
    return [SystemMessage(content=DEFAULT_SYSTEM_PROMPT)]


def generate_response(
    messages: list,
    model_name: str = "openai/gpt-oss-20b",
    context: str = None,
) -> str:
    """
    Takes the full conversation history (list of Message objects) and returns
    the AI's reply as a plain string.

    If `context` is provided (retrieved chunks from rag.py), the system prompt
    is swapped for RAG_SYSTEM_PROMPT so the model grounds its answer in the
    retrieved text instead of general knowledge.

    Note: we don't mutate the original SystemMessage in st.session_state —
    we build a fresh message list for this one call, so conversation memory
    stays clean regardless of whether RAG was used for a given turn.
    """
    llm = get_llm(model_name=model_name)

    # Filter out any existing system message rather than assuming messages[0]
    # is one — LangGraph nodes (Layer 4) may pass raw message lists that don't
    # guarantee a system message is first.
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    if context:
        system_msg = SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context))
    else:
        system_msg = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)

    call_messages = [system_msg] + non_system
    response = llm.invoke(call_messages)
    return response.content


# ---- Layer 3: Agent with tool-calling ----

AGENT_SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

Guidelines:
- If a document has been uploaded, use the doc_search tool FIRST for any question that might relate to it.
- Use web_search for general knowledge or current events not covered by the document.
- For ANY arithmetic or math calculation, no matter how simple, you MUST call the calculator tool.
  Do NOT compute the result yourself, even if you think you know the answer.
- If you can answer directly without any tool (e.g. a simple greeting), do so — don't call tools unnecessarily.
- Format your final answer in plain text only. Do NOT use LaTeX or markdown math notation of any kind
  (no backslashes, no special symbols for multiplication or grouping) — just write plain numbers and words.
- IMPORTANT: After calling a tool, your final answer MUST include the actual result/information the tool returned. Never just say "I used the X tool" without stating what it returned.
- Be concise in your final answer.
"""


def build_agent(tools: list, model_name: str = "openai/gpt-oss-20b"):
    """
    Builds a tool-calling agent using LangChain 1.x's native `create_agent`.
    Unlike generate_response() (Layer 1/2), this establishes an execution 
    loop letting the LLM decide to call zero or more tools, see their results, 
    and produce a final answer.
    """
    llm = get_llm(model_name=model_name)

    # Modern create_agent constructs its internal loop automatically from the model and tools
    agent = create_agent(
        model=llm,
        tools=tools,
    )
    return agent


def generate_agent_response(messages: list, tools: list, model_name: str = "openai/gpt-oss-20b"):
    """
    Runs the modern tool-calling agent on the conversation history.
    
    Returns (final_answer, intermediate_steps).
    """
    agent = build_agent(tools, model_name=model_name)

    # Robust to any message list shape: drop system messages
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    
    # Inject our modern AGENT_SYSTEM_PROMPT at the head of execution
    full_messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + non_system

    # invoke() natively processes the full message history, handles tool loops, 
    # and returns a final response under the 'messages' output key.
    result = agent.invoke({"messages": full_messages})
    
    # Extract the final AIMessage text content
    final_output = result["messages"][-1].content
    
    # Note: Modern create_agent structure returns complete message execution lists.
    # If your Layer 5 evaluation requires strict 'intermediate_steps' structures,
    # they are represented as AIMessages containing tool_calls in result["messages"].
    return final_output, result.get("messages", [])


def add_user_message(messages: list, user_input: str) -> list:
    messages.append(HumanMessage(content=user_input))
    return messages


def add_ai_message(messages: list, ai_response: str) -> list:
    messages.append(AIMessage(content=ai_response))
    return messages
