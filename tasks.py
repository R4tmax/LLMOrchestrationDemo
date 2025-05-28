# tasks.py
from crewai import Task

def create_tasks(query_analyst_agent, retrieval_agent, synthesizer_agent):
    analyse_query = Task(
        description=(
            "Analyze the user's input query: '{query}'. "
            "Identify the core intent, key entities, and any ambiguities. "
            "Output a refined query optimized for information retrieval from personal notes."
        ),
        expected_output="A single, clear, and optimized query string.",
        agent=query_analyst_agent
    )

    retrieve_info = Task(
        description=(
            "Using the refined query from the query analysis, search the Obsidian knowledge base. "
            "Retrieve the top 3 most relevant text chunks."
        ),
        expected_output="A formatted string containing the retrieved text chunks from Obsidian, including their sources. If nothing is found, state that clearly.",
        agent=retrieval_agent,
        context=[analyse_query] # Depends on the output of the first task
    )

    synthesize_answer = Task(
        description=(
            "Based on the original user query: '{query}' and the retrieved context from Obsidian notes, "
            "formulate a comprehensive and helpful answer. "
            "IMPORTANT: Base your answer *strictly* on the information present in the retrieved context. "
            "Do not use any external knowledge or make assumptions. "
            "If the context is insufficient to answer the query, clearly state that."
        ),
        expected_output="A well-structured and informative answer synthesized *only* from the provided Obsidian note chunks, directly addressing the user's query.",
        agent=synthesizer_agent,
        context=[retrieve_info] # Depends on the output of the retrieval task
    )
    return analyse_query, retrieve_info, synthesize_answer