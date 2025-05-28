# crew/tasks.py
from crewai import Task
from .agents import retriever_agent, answer_agent
from rag.retriever import get_retriever

def create_tasks(question: str):
    retriever = get_retriever()

    task1 = Task(
        description=f"Retrieve context from internal documents to answer: {question}",
        agent=retriever_agent,
        context=[question],  # ✅ Make this a list
        expected_output="A set of relevant context chunks extracted from internal documents."  # ✅ Add this
    )

    task2 = Task(
        description="Based on the retrieved context, write a complete answer.",
        agent=answer_agent,
        expected_output="A detailed, accurate, and helpful answer to the question."
    )

    return [task1, task2]
