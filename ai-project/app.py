"""
app.py
UI ONLY. This is now the single, unified interface — it doesn't know or care
whether a question was answered via RAG, a tool, or plain chat. All of that
routing lives in graph.py. app.py just: uploads a PDF (optional), sends user
messages to the graph, and renders whatever comes back.
"""

import streamlit as st
import os
import tempfile
import uuid

from rag import load_and_split_pdf, build_vectorstore, get_retriever
from graph import build_graph, seed_messages
from langchain_core.messages import HumanMessage, AIMessage

# ---- Page setup ----
st.set_page_config(page_title="Personal Knowledge Assistant", page_icon="🧠")
st.title("🧠 Personal Knowledge Assistant")
st.caption("RAG + tools + routing + memory — powered by a LangGraph pipeline")

# ---- Session state ----
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())  # unique per browser session — keys the checkpointer's memory

if "is_first_turn" not in st.session_state:
    st.session_state.is_first_turn = True

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "graph" not in st.session_state:
    st.session_state.graph = build_graph(retriever=None)  # no PDF yet


def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def get_history():
    """
    Pulls the full conversation from the graph's checkpointer — this IS the
    memory now, not a hand-managed list. Returns [] for a brand-new thread.
    """
    snapshot = st.session_state.graph.get_state(get_config())
    if not snapshot or not snapshot.values:
        return []
    return snapshot.values.get("messages", [])


# ---- Display chat history ----
for msg in get_history():
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    # SystemMessage intentionally not displayed

# ---- Chat input ----
user_input = st.chat_input("Ask me anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            input_messages = seed_messages(user_input, st.session_state.is_first_turn)
            result = st.session_state.graph.invoke(
                {"messages": input_messages},
                config=get_config(),
            )
            st.session_state.is_first_turn = False

            final_message = result["messages"][-1]
            st.markdown(final_message.content)

            # ---- Transparency panel: show HOW this turn was answered ----
            route = result.get("route")
            if route == "rag" and result.get("source_docs"):
                with st.expander("📄 Answered using your document"):
                    for doc in result["source_docs"]:
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**Page {page}:** {doc.page_content[:200]}...")
            elif route == "tool" and result.get("tool_steps"):
                with st.expander(f"🔧 Answered using tools ({len(result['tool_steps'])} call(s))"):
                    for action, observation in result["tool_steps"]:
                        st.markdown(f"**Tool:** `{action.tool}`  \n**Input:** `{action.tool_input}`")
                        st.markdown(f"**Result:** {str(observation)[:300]}")
                        st.divider()
            elif route == "chat":
                st.caption("💬 Answered from general knowledge — no document or tool needed")

# ---- Sidebar ----
with st.sidebar:
    st.header("📄 Document (optional)")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Reading, chunking, and embedding your PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = load_and_split_pdf(tmp_path)
            vectorstore = build_vectorstore(chunks)
            st.session_state.retriever = get_retriever(vectorstore)
            os.remove(tmp_path)

            # Rebuild the graph so its rag/tool nodes pick up the new retriever
            st.session_state.graph = build_graph(retriever=st.session_state.retriever)

        st.success(f"Processed {len(chunks)} chunks from '{uploaded_file.name}'.")

    if st.session_state.retriever:
        st.info("✅ Document loaded — I can answer from it, search the web, do math, or just chat.")
    else:
        st.caption("No document uploaded yet — I can still search the web, do math, or chat.")

    st.divider()
    if st.button("🗑️ New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())  # fresh thread = fresh memory
        st.session_state.is_first_turn = True
        st.rerun()

    st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")