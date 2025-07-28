from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI # langchain==1.0.0 


chat = ChatOpenAI(
    model="gpt-3.5-turbo",
)

# 会話履歴を含めてinvokeする
messages = [
    HumanMessage(content="茶碗蒸しの作り方を教えてください"),
    AIMessage(content="まず卵と出汁を混ぜて、具材と一緒に器に入れて蒸します。"),
    HumanMessage(content="英語に翻訳して"),
]

response = chat.invoke(messages)

print(response.content)