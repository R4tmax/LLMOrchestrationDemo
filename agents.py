from crewai import Agent
from tools.obsidian_retriever_tool import ObsidianSearchTool
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import BaseTool

class MyCustomDuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Search Tool"
    description: str = "Search the web for a given query."

    def _run(self, query: str) -> str:
        # Ensure the DuckDuckGoSearchRun is invoked properly.
        duckduckgo_tool = DuckDuckGoSearchRun()
        response = duckduckgo_tool.invoke(query)
        return response

    def _get_tool(self):
        # Create an instance of the tool when needed
        return MyCustomDuckDuckGoTool()


def create_agents(llm_instance):
    # Initialize the tools
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