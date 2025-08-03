import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda

from langchain.prompts import ChatPromptTemplate


llm = ChatOpenAI(
  model="gpt-3.5-turbo"
)

# 履歴（グローバルで保持）
conversation_history = []
# 最初の要約は空文字
summary_state = {"summary": ""}

response_prompt = ChatPromptTemplate.from_template(
    "これまでの会話の要約: {summary}\nユーザ: {input}\nアシスタント:"
)

summary_prompt = ChatPromptTemplate.from_template(
    "現在の要約:\n{existing_summary}\n新しい会話:\n{new_messages}\n\nこれらを基に新しい要約を返してください。"
)


def update_summary(existing_summary: str, new_messages: list):
    # メッセージの内容を1つの文字列に整形
    new_content = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in new_messages
    )
    return (
        summary_prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "existing_summary": existing_summary,
        "new_messages": new_content
    })


chatbot = response_prompt | llm | StrOutputParser()

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="私は会話の文脈を考慮した返答ができるチャットボットです。メッセージを入力してください。").send()

# メッセージ受信時の処理
@cl.on_message
async def on_message(message: cl.Message):
    print(f"ユーザーメッセージ: {message.content}")
    print(f"履歴件数: {len(conversation_history)}")

    # 1. ユーザー発言を履歴に追加
    conversation_history.append(HumanMessage(content=message.content))

    # 2. 要約を更新
    summary_state["summary"] = update_summary(
        summary_state["summary"],
        conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
    )

    # 3. 応答生成
    prompt_input = {
        "summary": summary_state["summary"],
        "input": message.content
    }
    print(prompt_input)

    result = await chatbot.ainvoke(prompt_input)

    # 4. 応答を履歴に追加
    conversation_history.append(AIMessage(content=result))

    # 5. クライアントに返答
    await cl.Message(content=result).send()