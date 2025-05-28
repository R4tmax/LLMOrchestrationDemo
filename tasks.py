# tasks.py
from crewai import Task

def create_tasks(query_analyst_agent, retrieval_agent, synthesizer_agent):
    analyse_query = Task(
        description=(
            "Analyze the user's input query: '{query}'. "
            "Identify the core intent, key entities, and any ambiguities. "
            "Output a refined query or a set of keywords optimized for information retrieval from personal notes." # Slightly adjusted
        ),
        expected_output=(
            "A concise and effective search query string or a list of key search terms "
            "derived from the user's original query. This will be used as input for a vector search tool."
        ),
        agent=query_analyst_agent
    )

    retrieve_info = Task(
        description=(
            "You have received a refined query/keywords:. " # {refined_query} will be the output from analyse_query task
            "Your primary and ONLY job in this task is to: "
            "1. If the refined_query is complex, extract the most relevant simple keyword(s) or phrase to use for a vector search. "
            "2. Use the 'Obsidian Vault Search' tool with this simple keyword/phrase. "
            "3. Your output for this task MUST be the direct, verbatim results returned by the 'Obsidian Vault Search' tool. "
            "Do NOT synthesize, summarize, or add any information not directly from the tool's output."
        ),
        expected_output=(
            "The exact text chunks and their sources as returned by the 'Obsidian Vault Search' tool. "
            "If the tool returns no results, you MUST output the exact phrase: "
            "'No relevant information found by the Obsidian Vault Search tool.' "
            "Do not add any other narrative or explanation."
        ),
        agent=retrieval_agent,
        context=[analyse_query]
    )

    synthesize_answer = Task(
        description=(
            "Based on the original user query: '{query}' and the retrieved context from Obsidian notes (output of the previous retrieval task), "
            "formulate a comprehensive and helpful answer. "
            "IMPORTANT: Base your answer *strictly* on the information present in the retrieved context. "
            "Do not use any external knowledge or make assumptions. "
            "If the context is insufficient or states that no information was found, clearly indicate that in your answer."
        ),
        expected_output="A well-structured and informative answer synthesized *only* from the provided Obsidian note chunks, directly addressing the user's query. If no information was found by the search tool, state that.",
        agent=synthesizer_agent,
        context=[retrieve_info, analyse_query] # Pass original query from analyse_query for context too
    )
    return analyse_query, retrieve_info, synthesize_answer