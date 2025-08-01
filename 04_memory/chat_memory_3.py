import os
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories  import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """Redis URLを使ってRedisChatMessageHistoryを作成"""
    return RedisChatMessageHistory(
        session_id=session_id,
        url=os.environ.get("REDIS_URL")
    )

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()
chatbot = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

### テスト実行 ###
# config = {"configurable": {"session_id": "user1"}}

# history = RedisChatMessageHistory(session_id="test-session", url=os.environ.get("REDIS_URL"))
# history.add_user_message("こんにちは")
# history.add_ai_message("こんにちは、田中さん！")
### テスト実行 終わり　###

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="私は会話の文脈を考慮した返答ができるちゃとっぼとです。メッセージを入力してください。").send()

@cl.on_message
async def on_message(message: cl.Message):
    # 会話モデル呼び出し（非同期）
    result = chatbot.invoke({"input": message.content}, config={"configurable": {"session_id": "user1"}})

    # クライアントへ応答
    await cl.Message(content=result).send()