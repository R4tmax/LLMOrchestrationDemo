# crew/crew.py
from crewai import Crew
from .agents import retriever_agent, answer_agent
from .tasks import create_tasks

def run_crew(question: str):
    tasks = create_tasks(question)
    crew = Crew(
        agents=[retriever_agent, answer_agent],
        tasks=tasks,
        verbose=True
    )
    return crew.kickoff()
