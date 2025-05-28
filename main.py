# main.py
from crew.crew import run_crew

if __name__ == "__main__":
    question = input("What would you like to ask about your notes? ")
    result = run_crew(question)
    print("\nFinal Answer:\n", result)
