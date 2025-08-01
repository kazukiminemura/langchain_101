import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="私は会話の文脈を考慮した返答ができるちゃとっぼとです。メッセージを入力してください。").send()

@cl.on_message
async def on_message(message: cl.Message):
    # メモリから履歴を取得（非同期）
    memory_message_result = memory.load_memory_variables({})
    chat_history = memory_message_result({})
    print(chat_history)

    # 最新のユーザー発言を履歴に追加
    chat_history.append(HumanMessage(content=message.content))

    # 会話モデル呼び出し（非同期）
    result = chat.invoke(chat_history)

    # 新しい履歴・応答を保存（非同期）
    memory.save_context(
        {
            "inputs": message.content,
        },
        {
            "outputs": result.content,
        }
    )

    # クライアントへ応答
    await cl.Message(content=result.content).send()