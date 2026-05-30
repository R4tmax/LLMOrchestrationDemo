"""
Task is an action required of the agent.
Note the official documentation https://docs.crewai.com/en/concepts/tasks.

In short, however, task is a template for the model to structure its behavior upon.
Think of agent definitions as of top level system prompts and of task definitions as personalities/subsetting
of your LLM (Gems in Gemini etc.)
This is where you want to prompt engineer to the best of your ability.
"""
from crewai import Task

def create_tasks(query_analyst_agent, retrieval_agent, synthesizer_agent):
    analyse_query_task = Task(
        description=(
            "Analyze the LATEST USER QUERY and CHAT HISTORY. "
            "Formulate a single, highly optimized search query for a vector database. "
            "Focus on core entities, nouns, and semantic meaning. Strip away conversational filler."
            "\n\nCHAT HISTORY:\n{chat_history}\n\nLATEST USER QUERY:\n{query}"
        ),
        expected_output="A single optimized search string (no extra text or explanations).",
        agent=query_analyst_agent
    )

    retrieve_info_task = Task(
        description=(
            "Using the optimized search string from the previous task, follow these EXACT steps:\n"
            "1. Execute the 'Obsidian Vault Search' tool using the search string.\n"
            "2. Read the returned text carefully. Does it actually contain information that answers the user's original query?\n"
            "3. If YES: Stop researching. Output the retrieved Obsidian text, prefixed with '[SOURCE: OBSIDIAN]'.\n"
            "4. If NO (or if the result is empty): Execute the 'DuckDuckGoSearch' tool to find the answer on the web.\n"
            "5. Output the web results, prefixed with '[SOURCE: WEB]'."
        ),
        expected_output="The raw text results from the successful tool, clearly prefixed with either [SOURCE: OBSIDIAN] or [SOURCE: WEB].",
        agent=retrieval_agent,
        context=[analyse_query_task]
    )

    synthesize_answer_task = Task(
        description=(
            "Synthesize a helpful, conversational answer for the user based strictly on the RETRIEVED CONTEXT from the previous task. "
            "1. If the context starts with [SOURCE: OBSIDIAN], treat it as the user's high-authority personal knowledge. "
            "2. If the context starts with [SOURCE: WEB], frame it as 'I couldn't find this in your notes, but I found this online...'. "
            "3. CRITICAL FALLBACK: If the retrieved context is empty, irrelevant, or states no information was found, DO NOT make up an answer. You MUST output a clear warning stating that the search was inconclusive across all available sources."
            "\n\nCHAT HISTORY:\n{chat_history}\n\nLATEST USER QUERY:\n{query}"
        ),
        expected_output="A natural answer citing the source, OR a clear warning that the system could not find conclusive information to answer the query.",
        agent=synthesizer_agent,
        context=[retrieve_info_task]
    )

    return analyse_query_task, retrieve_info_task, synthesize_answer_task