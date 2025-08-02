import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda

llm = ChatOpenAI(
  model="gpt-3.5-turbo"
)

# 履歴（グローバルで保持）
history = []

def trim_context_input(inputs: dict):
    # 会話履歴のリストから直近5件を抽出して新しい入力に追加
    trimmed = trim_messages(
        inputs["history"],
        token_counter=len,
        max_tokens=5,
        )
    history[:] = trimmed
    return trimmed + [HumanMessage(content=inputs["input"])]

chatbot = RunnableLambda(trim_context_input) | llm | StrOutputParser()

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="私は会話の文脈を考慮した返答ができるチャットボットです。メッセージを入力してください。").send()

@cl.on_message
async def on_message(message: cl.Message):
    print(f"ユーザーメッセージ: {message.content}")
    print(f"履歴: {len(history)}")

    # 会話モデル呼び出し（非同期）
    result = chatbot.invoke(
        {"input": message.content, "history": history},
        config={"configurable": {"session_id": "session_1"}}
    )

    # 履歴に追加（ユーザー・AIともに）
    history.append(HumanMessage(content=message.content))
    history.append(AIMessage(content=result))
  
    # クライアントへ応答
    await cl.Message(content=result).send()

    