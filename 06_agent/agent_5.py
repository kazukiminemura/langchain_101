from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.tools import WriteFileTool
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = []

retriever = WikipediaRetriever(
  lang="ja",
  doc_content_chars_max=500,
  top_k_results=1,
)

tools.append(
  create_retriever_tool(
    name="WikipediaRetriever",
    description="受けとった単語に関するWikipediaの記事を取得",
    retriever=retriever,
  )
)

memory = ConversationBufferMemory(
  memory_key="chat_history",
  return_messages=True,
)

agent = initialize_agent(
  tools=tools,
  llm=llm,
  agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
  memory=memory,
  verbose=True,
  allow_dangerous_tools=True,
)

result = agent.invoke("""スコッチウイスキーについてWikipediaで調べて概要を日本語でまとめてください。""")
print(f"一回目の実行結果: {result}")

result_2 = agent.invoke("""以前の指示をもう一度実行してください。""")
print(f"二回目の実行結果: {result}")