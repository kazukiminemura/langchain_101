from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = load_tools(["requests_get"], allow_dangerous_tools=True)
agent = initialize_agent(
  tools=tools,
  llm=llm,
  agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True,
  handle_parsing_errors=True,  # パースエラーを処理
)

result = agent.invoke("""以下のURLにアクセスして東京の天気をしらべて日本語で答えてください\
  https://www.jma.go.jp/bosai/forecast/data/overview_forecast/130000.json
  """)

print(f"実行結果: {result}")