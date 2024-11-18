from operator import itemgetter

import chainlit as cl
from chainlit.types import ThreadDict
from crewai import Agent, Crew, Process, Task
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain_ollama import ChatOllama

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    # if (username, password) == ("admin", "admin"):
    #     return cl.User(
    #         identifier="bitch", metadata={"role": "admin", "provider": "credentials"}
    #     )
    # else:
    #     return None
    return cl.User(identifier="bitch")

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
    description="Translate the paragraph from English to Vietnamese: {question}",
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

# async def async_crew_execution(input):
#     result = await crew.kickoff_async(inputs={"question":input})
#     # print("Crew Result:", result)
#     return result

@cl.on_chat_start
async def on_chat_start():
    app_user = cl.user_session.get("user")
    await cl.Message(f"Hello {app_user.identifier}!").send()
    
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    # start_chat()

@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")
    # runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    # async for chunk in runnable.astream(
    #     {"question": message.content},
    #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await msg.stream_token(chunk)
    # await msg.stream_token(runnable.send({"question":message.content}))

    print("start finding ans")
    result = await crew.kickoff_async(inputs={"question":message.content})
    await msg.stream_token(result.raw)

    await msg.send()
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)
