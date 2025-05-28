from crewai import Agent
from rag.hf_llm import query_hf

retriever_agent = Agent(
    role="Retriever",
    goal="Find relevant information from internal notes",
    backstory="An expert researcher that knows how to find key details.",
    verbose=True,
    llm=query_hf
)

answer_agent = Agent(
    role="Answer Generator",
    goal="Generate helpful answers based on retrieved knowledge",
    backstory="An articulate AI with deep understanding of the subject matter.",
    verbose=True,
    llm=query_hf
)
