from langchain.agents import AgentType, initialize_agent, Tool
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.tools import WriteFileTool

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = []
tools.append(WriteFileTool(root_dir="./"))

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

agent = initialize_agent(
  tools=tools,
  llm=llm,
  agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True,
  allow_dangerous_tools=True,
)

result = agent.invoke("""スコッチウイスキーについてWikipediaで調べて概要を日本語でresult.txtというファイルに保存してください。""")

print(f"実行結果: {result}")
