# To be used in agents.py or main.py when defining agents
from crewai import Crew, Process
from agents import create_agents
from tasks import create_tasks
from config import HF_INFERENCE_API_URL, HF_API_TOKEN, OBSIDIAN_VAULT_PATH, VECTOR_STORE_INDEX_PATH, \
    VECTOR_STORE_DOCS_PATH
from obsidian_processor import setup_knowledge_base
from langchain_community.llms import HuggingFaceEndpoint
import os


# Ensure HF_API_TOKEN is in environment if HuggingFaceEndpoint relies on it by default
# Or pass it directly if the class supports it (check Langchain docs for HuggingFaceEndpoint)
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_TOKEN # HuggingFaceEndpoint often picks this up

online_llm = HuggingFaceEndpoint(
    endpoint_url=HF_INFERENCE_API_URL,
    task="text-generation", # Common task for instruction-following models
    # You might need to add model_kwargs for specific controls like temperature, max_new_tokens
    # e.g., model_kwargs={"temperature": 0.7, "max_new_tokens": 500}
    huggingfacehub_api_token = HF_API_TOKEN # Explicitly pass token
)

# main.py


def run_demo():
    # --- LLM Initialization ---
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_TOKEN  # Ensure token is set
    online_llm = HuggingFaceEndpoint(
        endpoint_url=HF_INFERENCE_API_URL,
        task="text-generation",
        # model_kwargs={"temperature": 0.2, "max_new_tokens": 1024}, # Example: adjust for desired output
        huggingfacehub_api_token=HF_API_TOKEN
    )
    print("Hugging Face Online LLM Initialized.")

    # --- Agent and Task Initialization ---
    query_analyst, retrieval_specialist, notes_synthesizer = create_agents(online_llm)
    print("Agents Created.")

    analyse_query_task, retrieve_info_task, synthesize_answer_task = create_tasks(
        query_analyst, retrieval_specialist, notes_synthesizer
    )
    print("Tasks Created.")

    # --- Crew Definition ---
    obsidian_crew = Crew(
        agents=[query_analyst, retrieval_specialist, notes_synthesizer],
        tasks=[analyse_query_task, retrieve_info_task, synthesize_answer_task],
        process=Process.sequential,  # Can also be hierarchical
        verbose=True  # For detailed output
    )
    print("Crew Defined.")

    # --- Get User Query ---
    user_query = input("Ask a question about your Obsidian notes: ")
    if not user_query:
        print("No query entered. Exiting.")
        return

    inputs = {'query': user_query}

    # --- Kick off the Crew ---
    print("\nKicking off the crew...\n")
    result = obsidian_crew.kickoff(inputs=inputs)

    print("\n\n########################")
    print("##### Crew Result ######")
    print("########################\n")
    print(result)


if __name__ == "__main__":
    # --- Initial Setup (run once or if notes change) ---
    # Check if knowledge base exists, otherwise create it
    if not (os.path.exists(VECTOR_STORE_INDEX_PATH) and os.path.exists(VECTOR_STORE_DOCS_PATH)):
        print("Knowledge base not found. Setting up...")
        setup_knowledge_base(OBSIDIAN_VAULT_PATH, VECTOR_STORE_INDEX_PATH, VECTOR_STORE_DOCS_PATH)
    else:
        print("Knowledge base found.")

    run_demo()