from crewai import Task


def create_tasks(query_analyst_agent, retrieval_agent, synthesizer_agent):
    analyse_query_task = Task(
        description=(
            "Analyze the LATEST USER QUERY and CHAT HISTORY. "
            "1. Formulate a search query optimized for vector search (keywords). "
            "2. Determine if this topic likely exists in the 'Personal Obsidian Notes' or requires 'External Web Search'. "
            "   (Hint: Personal meetings, projects, and private thoughts are Obsidian. News, generic definitions, and libraries are Web). "
            "\n\nCHAT HISTORY:\n{chat_history}\n\nLATEST USER QUERY:\n{query}"
        ),
        expected_output="A refined search query and a brief suggestion on which source (Obsidian or Web) is most likely to have the answer.",
        agent=query_analyst_agent
    )

    # note, LLM semantically struggles with decision process on the first step
    retrieve_info_task = Task(
        description=(
            "You have a refined query and a source suggestion. "
            "EXECUTION STEPS: "
            "1. ALWAYS PRIORITISE trying the 'Obsidian Vault Search' tool first using the refined query.  "
            "2. Analyze the results. If the Obsidian search returns 'No relevant information found' or irrelevant text: "
            "3. IMMEDIATELY use the 'DuckDuckGoSearch' tool to find the answer on the web. "
            "4. If you used the Web, make sure to verify the credibility of the source briefly."
        ),
        expected_output=(
            "The raw text results from the tool used. "
            "Prefix the output with '[SOURCE: OBSIDIAN]' or '[SOURCE: WEB]' so the next agent knows where it came from."
        ),
        agent=retrieval_agent,
        context=[analyse_query_task]
    )

    synthesize_answer_task = Task(
        description=(
            "Synthesize a helpful answer for the user based on the RETRIEVED CONTEXT. "
            "1. If the context starts with [SOURCE: OBSIDIAN], treat it as high-authority personal knowledge. "
            "2. If the context starts with [SOURCE: WEB], frame it as 'I found this online...'. "
            "3. If no information was found in either, apologize politely. "
            "\n\nCHAT HISTORY:\n{chat_history}\n\nLATEST USER QUERY:\n{query}"
        ),
        expected_output="A natural, conversational answer that explicitly references whether the info came from personal notes or the internet.",
        agent=synthesizer_agent,
        context=[retrieve_info_task, analyse_query_task]
    )

    return analyse_query_task, retrieve_info_task, synthesize_answer_task