import time
import langchain
from langchain_community.cache import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# langchainのキャッシュを有効にする
langchain.llm_cache = InMemoryCache()


chat = ChatOpenAI()
start = time.time()
result = chat.invoke([
    HumanMessage(content="こんにちは!")
])
end = time.time()

print(result.content)
print(f"実行時間: {end - start}秒")

start = time.time()
# 再度同じメッセージを送信
result = chat.invoke([
    HumanMessage(content="こんにちは!")
])
end = time.time()

print(result.content)
print(f"実行時間: {end - start}秒")

