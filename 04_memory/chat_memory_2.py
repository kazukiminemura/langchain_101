import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOpenAI(model="gpt-3.5-turbo")

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain = RunnableWithMessageHistory(llm, get_session_history)


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="私は会話の文脈を考慮した返答ができるちゃとっぼとです。メッセージを入力してください。").send()

@cl.on_message
async def on_message(message: cl.Message):
    # 会話モデル呼び出し（非同期）
    result = chain.invoke(message.content, config={"configurable": {"session_id": "user1"}})

    # クライアントへ応答
    await cl.Message(content=result.content).send()