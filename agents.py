"""
agents script is responsible for preparing the objects for CrewAI agents.
See https://docs.crewai.com/en/concepts/agents for conceptual definition in framework.

For all practical intents and purposes, agent is the top level archetype of the agentic system.
It is the agent - in our case powered by the LLM, but in theory it can be any piece of software based logic capable of
performing conditional logic - who is responsible for receiving input, process it, make a decision and provide output.

Tool is an action outside the capability of the agent, It can be thought of both as an auxiliary or primary method of the agentic action.
Since we typically think about agentic systems as LLMs based decision frameworks, and as such in theory you can have a system completely void of tools.
Inversely you could have a system, which basically only chains tools.
Degree of freedom of thought in the agentic system is largely dependent on the prompt engineering/finetuning done by you as the developer
at the orchestration level.
"""

from crewai import Agent
from tools.obsidian_retriever_tool import ObsidianSearchTool
from tools.duckDuckGo_tool import MyCustomDuckDuckGoTool

# Called from main to prepare the "crew"
def create_agents(llm_instance):
    # Instantiate the tool objects
    obsidian_tool = ObsidianSearchTool()
    web_search_tool = MyCustomDuckDuckGoTool()


    query_analyst = Agent(
        role='Expert Query Analyst',
        goal='Refine user queries to be highly effective. Decide if the query could be answered using internal knowledge stored in a "digital brain" (Obsidian), or external knowledge (Web). Assume the context of Software engineering formal education on part of the creator of the Obsidian Vault with a collection of esoteric interests. Only advice external search as a last resort (e.g. information is outside of suspected domains and/or unreasonably specific)',
        backstory='You are a master at understanding user intent. You analyze whether a user is asking about personal notes or general world knowledge.',
        verbose=True,
        allow_delegation=False,
        llm=llm_instance
    )

    retrieval_specialist = Agent(
        role='Information Research Specialist',
        goal='Find the best answer using EITHER the Obsidian Vault OR the Web.',
        backstory=(
            "You are a versatile researcher. Your primary source of truth is the Obsidian Vault. "
            "However, if the user asks for recent events, general facts not in the notes, or if "
            "the Obsidian search returns nothing, you are authorized to search the Web. "
            "ALWAYS PRIORITISE trying Obsidian first, unless previous suggestions advise not to do so."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[obsidian_tool, web_search_tool],
        llm=llm_instance
    )

    notes_synthesizer = Agent(
        role='Insightful Content Synthesizer',
        goal='Synthesize a comprehensive answer based on the retrieved context (whether internal or external).',
        backstory='You piece together information from various sources to construct clear answers. You explicitly state where the information came from (Notes vs. Web).',
        verbose=True,
        allow_delegation=False,
        llm=llm_instance
    )

    return query_analyst, retrieval_specialist, notes_synthesizer