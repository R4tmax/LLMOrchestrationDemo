# To be used in agents.py or main.py when defining agents
from crewai import Crew, Process
from agents import create_agents
from tasks import create_tasks
from config import HF_MODEL_ID, HF_API_TOKEN, OBSIDIAN_VAULT_PATH, VECTOR_STORE_INDEX_PATH, \
    VECTOR_STORE_DOCS_PATH
from obsidian_processor import setup_knowledge_base
from langchain_huggingface import HuggingFaceEndpoint
import os
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch # If using transformers
from crewai import LLM


def run_demo():
    # --- LLM Initialization ---
    # Ensure HF_API_TOKEN is in environment if HuggingFaceEndpoint relies on it by default
    # Or pass it directly if the class supports it (check Langchain docs for HuggingFaceEndpoint)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_TOKEN  # HuggingFaceEndpoint often picks this up
    os.environ["HF_TOKEN"] = HF_API_TOKEN
    #print(HF_API_TOKEN)
    #print(HF_MODEL_ID)

    llm = LLM(
        model=f"huggingface/{HF_MODEL_ID}"
    )
    print("LLM Initialized.")

    # main.py

    # --- Agent and Task Initialization ---
    query_analyst, retrieval_specialist, notes_synthesizer = create_agents(llm)
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