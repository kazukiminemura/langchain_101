import os
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories  import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage

def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """Redis URLを使ってRedisChatMessageHistoryを作成"""
    return RedisChatMessageHistory(
        session_id=session_id,
        url=os.environ.get("REDIS_URL")
    )

@cl.on_chat_start
async def on_chat_start():
    thread_id = None
    while not thread_id:
        res = await cl.AskUserMessage(
            content="私は会話の文脈を考慮した返答ができるちゃとっぼとです。スレッドIDを入力してください", timeout=600).send()
        if res:
            thread_id = res["output"]
            print(f"Received thread_id: {thread_id}")
        
    # LLMチェーン作成
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    # RunnableWithMessageHistory に履歴管理関数を登録
    chatbot = RunnableWithMessageHistory(
        runnable=llm | StrOutputParser(),
        get_session_history=get_session_history,
    )

    # 過去のメッセージを表示
    history = get_session_history(thread_id)
    for msg in history.messages:
        author = "User" if isinstance(msg, HumanMessage) else "Chatbot"
        await cl.Message(author=author, content=msg.content).send()
        
    cl.user_session.set("chatbot", chatbot)
    cl.user_session.set("thread_id", thread_id)

@cl.on_message
async def on_message(message: cl.Message):
    chatbot = cl.user_session.get("chatbot")
    thread_id = cl.user_session.get("thread_id")
    # 会話モデル呼び出し（非同期）
    result = chatbot.invoke(
        message.content,
        config={"configurable": {"session_id": thread_id}}
    )

    # クライアントへ応答
    await cl.Message(content=result).send()