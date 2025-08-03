from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import WriteFileTool

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = load_tools(["requests_get", "serpapi"], llm=llm, allow_dangerous_tools=True)
tools.append(WriteFileTool(root_dir="./"))

agent = initialize_agent(
  tools=tools,
  llm=llm,
  agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True,
  allow_dangerous_tools=True,
)

result = agent.invoke("""北海道の名産品を調べて日本語でresult.txtというファイルに保存してください。
  """)

print(f"実行結果: {result}")
