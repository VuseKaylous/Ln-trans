import asyncio
from crewai import Crew, Agent, Task, Process
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")

# Create an agent with code execution enabled
translator = Agent(
    role="Translator",
    goal="Translate the document from English to Vietnamese",
    backstory="You are an expert translator specialized in translating from English to Vietnamese. You are currently tasked to translate the given passage.",
    allow_delegation=False,
    llm=llm,
)

# Create a task that requires code execution
translate_task = Task(
    description="Translate the paragraph from English to Vietnamese: {input}",
    expected_output="A Vietnamese paragraph.",
    agent=translator,
)

# Create a crew and add the task
crew = Crew(
    agents=[translator],
    tasks=[translate_task],
    # manager_agent=manager,
    process=Process.sequential,
)
# Async function to kickoff the crew asynchronously
async def async_crew_execution():
    result = await crew.kickoff_async(inputs={"input":"Good morning America"})
    print("Crew Result:", result)

# Run the async function
asyncio.run(async_crew_execution())
