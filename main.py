import streamlit as st
from crewai import Crew, Process, LLM
from agents import create_agents  # Assuming these are in agents.py
from tasks import create_tasks  # Assuming these are in tasks.py
from config import GEMINI_API_KEY, OBSIDIAN_VAULT_PATH, VECTOR_STORE_INDEX_PATH, VECTOR_STORE_DOCS_PATH
from obsidian_processor import setup_knowledge_base
import os

# --- Page Configuration (Good practice for Streamlit apps) ---
st.set_page_config(page_title="Obsidian RAG Chat", layout="wide")


# --- LLM Initialization (Cached to run once) ---
@st.cache_resource  # Cache the LLM resource to avoid re-initializing on every interaction
def initialize_llm():
    # Ensure GEMINI_API_KEY is correctly picked up by LiteLLM.
    # LiteLLM typically expects os.environ["GEMINI_API_KEY"] or passes it in llm_params.
    # Setting it globally here for LiteLLM to pick up if crewai.LLM doesn't pass it directly.
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    # Verify the correct model string for Gemini Flash via LiteLLM.
    # Common options: "gemini/gemini-1.5-flash-latest" or just "gemini-1.5-flash-latest"
    # "gemini/gemini-2.0-flash" might not be a recognized LiteLLM string.
    # Let's use a known valid one:
    llm_instance = LLM(model=f"gemini/gemini-2.0-flash")  # Using crewai.LLM which uses LiteLLM
    print("LLM Initialized for Streamlit app.")
    return llm_instance


llm = initialize_llm()


# --- Agent Initialization (Cached) ---
@st.cache_resource  # Agents can also be cached as they depend on the LLM
def get_agents(_llm):
    query_analyst, retrieval_specialist, notes_synthesizer = create_agents(_llm)
    print("Agents Created for Streamlit app.")
    return query_analyst, retrieval_specialist, notes_synthesizer


query_analyst, retrieval_specialist, notes_synthesizer = get_agents(llm)

# --- Knowledge Base Setup (Run once per app session) ---
if 'kb_setup_done' not in st.session_state:
    if not (os.path.exists(VECTOR_STORE_INDEX_PATH) and os.path.exists(VECTOR_STORE_DOCS_PATH)):
        with st.spinner("Setting up Obsidian knowledge base... This may take a moment."):
            setup_knowledge_base(OBSIDIAN_VAULT_PATH, VECTOR_STORE_INDEX_PATH, VECTOR_STORE_DOCS_PATH)
        st.sidebar.success("Knowledge base setup complete!")
    else:
        st.sidebar.info("Knowledge base found.")
    st.session_state.kb_setup_done = True

# --- Streamlit UI ---
st.title(" 🗣️ Chat with your Obsidian Notes")
st.caption("Powered by CrewAI & Gemini")

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
        message_placeholder = st.empty()  # Used for streaming-like effect if needed, or just to show thinking
        message_placeholder.markdown("Thinking...")

        # Construct chat history string for CrewAI input
        # We'll pass the last few messages to avoid overly long context for the crew, adjust as needed
        # For a more robust solution, a proper summarization or selection strategy for history might be needed.
        history_for_crew = []
        for msg in st.session_state.messages[:-1]:  # All messages except the current user_prompt
            history_for_crew.append(f"{msg['role']}: {msg['content']}")
        chat_history_str = "\n---\n".join(history_for_crew)

        # Dynamically create tasks with the current context.
        # Your tasks.py's create_tasks function should define tasks whose descriptions
        # can accept {query} and {chat_history} from the inputs dict.
        current_tasks = create_tasks(query_analyst, retrieval_specialist, notes_synthesizer)

        inputs_for_crew = {
            'query': user_prompt,  # The current user question
            'chat_history': chat_history_str  # The history leading up to this question
        }

        # Create and run the crew for this turn
        obsidian_crew = Crew(
            agents=[query_analyst, retrieval_specialist, notes_synthesizer],
            tasks=current_tasks,
            process=Process.sequential,
            verbose=True  # Keep False for cleaner Streamlit UI; True for console debugging
            # memory=True # CrewAI's memory could be an alternative, but explicit history is often clearer for RAG.
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