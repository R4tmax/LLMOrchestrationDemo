"""
This is the entrypoint of the whole application.

There are comments throughout to guide you through.
TLDR. we instantiate a bunch of crewAI objects and structure them in a streamlit UI.
LLM part of the application is fueled by RPCs to Gemini via GCP Vertex API keys.

If this is your first interaction with LLM orchestration and agentic systems, be sure to work alongside the docs
(https://docs.crewai.com/en/introduction) to cover your bases.
Personally, I don't have strong opinions about what library/framework works best. I use crewAI here because
I have some prior experience with it, and I have the ENV setup on my laptop. In general I would say that
staying as close to the langGraph ecosystem as you can is the smartest approach (https://docs.langchain.com/),
its the most used and it is what Dr. Vencovský knows best, and as such can help with.

I you are still struggling with what an LLM is and how it works, I recommend this YT series:
https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
If you understand that, you know everything you will ever need to know on the conceptual level.
"""

import streamlit as st
from crewai import Crew, Process, LLM
from agents import create_agents
from tasks import create_tasks
from config import GEMINI_API_KEY, OBSIDIAN_VAULT_PATH, VECTOR_STORE_INDEX_PATH, VECTOR_STORE_DOCS_PATH
from obsidian_processor import setup_knowledge_base
import os

from visualize_vectors import render_vector_space
from visualize_vectors2d import render_vector_space2d

# --- Page Configuration ---
st.set_page_config(page_title="Obsidian RAG Chat", layout="wide")


# --- LLM Initialization  ---
@st.cache_resource  # Cache the LLM resource to avoid re-initializing on every interaction
def initialize_llm():
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    # take note that the system is not conforming to User/Agent flow (User-agent-agent-agent-user)
    # Certain models might struggle with activating and using the flow
    # Inability of model to conform to the flow will be noted by the resulting error
    llm_instance = LLM(model="gemini-pro-latest")
    print("LLM Initialized for Streamlit app.")
    return llm_instance


llm = initialize_llm()
# --- Knowledge Base Setup ---
if 'kb_setup_done' not in st.session_state:
    if not (os.path.exists(VECTOR_STORE_INDEX_PATH) and os.path.exists(VECTOR_STORE_DOCS_PATH)):
        with st.spinner("Setting up Obsidian knowledge base... This may take a moment."):
            setup_knowledge_base(OBSIDIAN_VAULT_PATH, VECTOR_STORE_INDEX_PATH, VECTOR_STORE_DOCS_PATH)
        st.sidebar.success("Knowledge base setup complete!")
    else:
        st.sidebar.info("Knowledge base found.")
    st.session_state.kb_setup_done = True

# this server runs prepared gemma variant for student use
#https://deeplearning.vse.cz:80
#ollama/gemma3:4b-it-qat

# --- Agent Initialization  ---
@st.cache_resource
def get_agents(_llm):
    query_analyst, retrieval_specialist, notes_synthesizer = create_agents(_llm)
    print("Agents Created for Streamlit app.")
    return query_analyst, retrieval_specialist, notes_synthesizer


query_analyst, retrieval_specialist, notes_synthesizer = get_agents(llm)



# --- Streamlit UI ---
st.title(" Obsidian Vault Assistant")

with st.sidebar.expander("🗺️ View Vector Map (3D)"):
    render_vector_space()

with st.sidebar.expander("🗺️ View Vector Map (2D)"):
    render_vector_space2d()



# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_prompt := st.chat_input("Ask a question about your notes..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Prepare for AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        # Naive history collection for the chat history
        # think of this as conversation context
        history_for_crew = []
        for msg in st.session_state.messages[:-1]:  # All messages except the current user_prompt
            history_for_crew.append(f"{msg['role']}: {msg['content']}")
        chat_history_str = "\n---\n".join(history_for_crew)

        current_tasks = create_tasks(query_analyst, retrieval_specialist, notes_synthesizer)

        inputs_for_crew = {
            'query': user_prompt,  # The current user question
            'chat_history': chat_history_str  # The history leading up to this question
        }

        # Create and run the crew for this turn, flow in this case is sequential for instructive reasons
        # note that the Crew/Multiagent system behavior does not and it a lot of cases should not be like so.
        obsidian_crew = Crew(
            agents=[query_analyst, retrieval_specialist, notes_synthesizer],
            tasks=current_tasks,
            process=Process.sequential,
            verbose=True  # I use this to be more instructive, this affects the level of detail in your CLI prints
            # memory=True  # I handle my own context window
        )

        try:
            # Kick off the crew with the current query and chat history
            with st.spinner("CrewAI is processing your request..."):
                result = obsidian_crew.kickoff(inputs=inputs_for_crew)

            ai_response = result
            message_placeholder.markdown(ai_response)

        except Exception as e:
            ai_response = f"An error occurred: {str(e)}"
            st.error(ai_response)
            import traceback

            print("Error details:")  # Log to console
            traceback.print_exc()

    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})