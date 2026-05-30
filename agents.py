"""
agents script is responsible for preparing the objects for CrewAI agents.
See https://docs.crewai.com/en/concepts/agents for conceptual definition in framework.

For all practical intents and purposes, agent is the top level archetype of the agentic system.
It is the agent who is responsible for receiving input, process it, make a decision and provide output. Note that different Orch. frameworks might
have different default behaviors for undefined LLMs, my recommendation is to always define one and not overcomplicate it.
In theory, you could have agentic/agent-like systems using non LLM decision nodes, for the purposes of this subject I advise caution,
do not overcomplicate it needlessly -> in most cases easier solution is to duplicate the LLM usage and use a tool to provide a different logic execution procedure.

Tool is an action outside the standard capability of the agent, It can be thought of both as an auxiliary or primary method of the agentic action.
Since we typically think about agentic systems as LLMs based decision frameworks, and as such in theory you can have a system completely void of tools.
Inversely you could have a system, which basically only chains tools.
Degree of freedom of thought in the agentic system is largely dependent on the prompt engineering/finetuning done by you as the developer
at the orchestration level.
"""

from crewai import Agent
from tools.obsidian_retriever_tool import ObsidianSearchTool
from tools.duckDuckGo_tool import MyCustomDuckDuckGoTool

def create_agents(llm_instance):
    obsidian_tool = ObsidianSearchTool()
    web_search_tool = MyCustomDuckDuckGoTool()

    query_analyst = Agent(
        role='Expert Query Analyst',
        goal='Refine user queries into highly effective, keyword-dense search strings optimized for vector database retrieval.',
        backstory='You are a master at understanding user intent and extracting semantic keywords. You do not answer the question; you only translate human conversation into optimal search queries.',
        verbose=True,
        allow_delegation=False,
        llm=llm_instance
    )

    retrieval_specialist = Agent(
        role='Information Research Specialist',
        goal='Retrieve relevant information to answer the user query by STRICTLY searching the Obsidian Vault first, and falling back to the Web ONLY if the vault lacks the answer.',
        backstory=(
            "You are a methodical researcher. You must ALWAYS use the Obsidian tool first. "
            "Once you get the Obsidian results, you evaluate them. If they contain the answer, your job is done. "
            "If they are empty or irrelevant to the actual question, you are authorized to pivot and use the Web search tool."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[obsidian_tool, web_search_tool],
        llm=llm_instance
    )

    notes_synthesizer = Agent(
        role='Insightful Content Synthesizer',
        goal='Synthesize a comprehensive answer based ONLY on the retrieved context. If the context is inconclusive, explicitly warn the user instead of answering.',
        backstory=(
            "You are a clear and honest communicator. You piece together information from the provided context. "
            "You always explicitly state whether your answer is based on personal notes or web search results. "
            "CRUCIALLY, if the researcher provides empty, irrelevant, or inconclusive results, you refuse to guess. "
            "Instead, you clearly warn the user that the information is missing from both the vault and the web."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm_instance
    )

    return query_analyst, retrieval_specialist, notes_synthesizer