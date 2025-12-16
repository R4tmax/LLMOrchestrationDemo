"""
Tool is available as pypi package, but it needs to be wrapped in BaseTool Instance to comply with crewAI framework.
https://github.com/deedy5/ddgs
"""
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import BaseTool

class MyCustomDuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Search Tool"
    description: str = "Search the web for a given query."

    def _run(self, query: str) -> str:
        duckduckgo_tool = DuckDuckGoSearchRun()
        response = duckduckgo_tool.invoke(query)
        return response

    def _get_tool(self):
        return MyCustomDuckDuckGoTool()