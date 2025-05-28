# agents.py
from crewai import Agent
from tools.obsidian_retriever_tool import ObsidianSearchTool
# online_llm should be initialized as shown above (e.g., in main.py and passed or initialized here)

# Example: Assuming online_llm is initialized elsewhere and passed or imported
# from main import online_llm # If initialized in main

def create_agents(llm_instance): # Pass the initialized LLM
    query_analyst = Agent(
        role='Expert Obsidian Query Analyst',
        goal='Refine user queries to be highly effective for searching a personal Obsidian knowledge base. Identify key concepts and entities.',
        backstory='You are a master at understanding user intent and translating it into precise, searchable queries for a knowledge management system like Obsidian.',
        verbose=True,
        allow_delegation=False,
        llm=llm_instance
    )

    retrieval_specialist = Agent(
        role='Obsidian Knowledge Retriever',
        goal='Use the refined query to search the Obsidian vector knowledge base and retrieve the most relevant text chunks.',
        backstory='You are a specialist in navigating vectorized information stores. Your mission is to find the exact pieces of information needed, based on the provided query.',
        verbose=True,
        allow_delegation=False,
        tools=[ObsidianSearchTool()],
        llm=llm_instance
    )

    notes_synthesizer = Agent(
        role='Insightful Content Synthesizer for Obsidian Notes',
        goal='Analyze the retrieved Obsidian note chunks in context of the original query and synthesize a comprehensive, coherent, and accurate answer. Base your answer *only* on the provided context.',
        backstory='You excel at piecing together information from various text snippets to construct clear answers. You are careful to only use the information given from the Obsidian notes.',
        verbose=True,
        allow_delegation=False,
        llm=llm_instance
    )
    return query_analyst, retrieval_specialist, notes_synthesizer