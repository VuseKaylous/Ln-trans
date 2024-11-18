from operator import itemgetter
# from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig

from langchain.memory import ConversationBufferMemory

from chainlit.types import ThreadDict
import chainlit as cl

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

def start_chat():
    memory = cl.user_session.get("memory")
    model = ChatOllama(model="llama3")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a professional translator and is handling a job that requires you to translate the given text from English to Vietnamese.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        ) | prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_chat_start
async def on_chat_start():
    app_user = cl.user_session.get("user")
    await cl.Message(f"Hello {app_user.identifier}!").send()
    
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    start_chat()
    

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

    start_chat()

@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        # print("Chunk: " + chunk)
        await msg.stream_token(chunk)

    await msg.send()
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)

