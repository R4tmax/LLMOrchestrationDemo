# tasks.py
from crewai import Task

def create_tasks(query_analyst_agent, retrieval_agent, synthesizer_agent):
    # The placeholders {query} and {chat_history} will be filled from the 'inputs' dictionary
    # provided to the crew.kickoff() method.

    analyse_query_task = Task(
        description=(
            "Given the CHAT HISTORY and the LATEST USER QUERY, your goal is to understand the user's current information need. "
            "If the LATEST USER QUERY is a follow-up, use the CHAT HISTORY to resolve ambiguities and understand the context. "
            "Based on this understanding, formulate a concise and effective search query or a list of keywords. "
            "This output will be used to search a personal Obsidian knowledge base. "
            "Focus on extracting key entities and concepts from the LATEST USER QUERY, informed by the CHAT HISTORY. "
            "\n\nCHAT HISTORY:\n{chat_history}\n\nLATEST USER QUERY:\n{query}"
        ),
        expected_output=(
            "A search query string or a list of keywords optimized for vector search against Obsidian notes, "
            "reflecting the user's latest information need derived from the LATEST USER QUERY and CHAT HISTORY."
        ),
        agent=query_analyst_agent
    )

    retrieve_info_task = Task(
        description=(
            "You have received a 'REFINED SEARCH QUERY' (output from the previous analysis step). "
            "Your ONLY job is to: "
            "1. If the 'REFINED SEARCH QUERY' is complex (e.g., a long sentence or multiple keywords), identify the most critical part or simplify it into a concise search phrase. "
            "2. Use the 'Obsidian Vault Search' tool with this focused search phrase. "
            "3. Your output MUST be the direct, verbatim results returned by the 'Obsidian Vault Search' tool. "
            "Do NOT add any information not directly from the tool's output. "
            "\n\n(The REFINED SEARCH QUERY will be automatically passed from the previous task's output)."
        ),
        expected_output=(
            "The exact text chunks and their sources as returned by the 'Obsidian Vault Search' tool. "
            "If the tool finds nothing, you MUST output the exact phrase: "
            "'No relevant information found by the Obsidian Vault Search tool.'"
        ),
        agent=retrieval_agent,
        context=[analyse_query_task] # Output of analyse_query_task is implicitly passed as {refined_query} or similar
    )

    synthesize_answer_task = Task(
        description=(
            "Your task is to synthesize a conversational and helpful answer for the user. "
            "Base your answer STRICTLY on the information present in the 'RETRIEVED OBSIDIAN CONTEXT' (output from the search tool) "
            "and the 'LATEST USER QUERY'. "
            "Use the 'CHAT HISTORY' to maintain conversational flow, understand the user's ongoing needs, and make your response natural. "
            "Do NOT use any external knowledge. "
            "If the 'RETRIEVED OBSIDIAN CONTEXT' indicates no information was found or is insufficient, "
            "clearly state that in your answer, but still try to be conversationally helpful based on the CHAT HISTORY and LATEST USER QUERY. "
            "\n\nCHAT HISTORY:\n{chat_history}\n\nLATEST USER QUERY:\n{query}\n\n(The RETRIEVED OBSIDIAN CONTEXT will be automatically passed from the previous task's output)."
        ),
        expected_output=(
            "A well-structured, natural-sounding, and informative answer that directly addresses the LATEST USER QUERY. "
            "The answer must be synthesized *only* from the RETRIEVED OBSIDIAN CONTEXT, considering the CHAT HISTORY for tone and context. "
            "If no relevant information was found by the search tool, state that clearly and politely."
        ),
        agent=synthesizer_agent,
        context=[retrieve_info_task, analyse_query_task] # Provides retrieve_info output & original query context
    )
    return analyse_query_task, retrieve_info_task, synthesize_answer_task