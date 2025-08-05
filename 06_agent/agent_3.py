import random
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import WriteFileTool

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
#tools = load_tools(["requests_get", "serpapi"], llm=llm, allow_dangerous_tools=True)
tools = []
tools.append(WriteFileTool(root_dir="./"))

def min_limit_random_number(min_number):
  return random.randint(int(min_number), 100000)

tools.append(Tool(
  name="Random",
  func=min_limit_random_number,
  description="特定の最小値以上のランダムな数値を生成します。"
))

agent = initialize_agent(
  tools=tools,
  llm=llm,
  agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True,
  allow_dangerous_tools=True,
)

result = agent.invoke("""１０以上のランダムな数字を生成してrandom.txtというファイルに保存してください。`
  """)

print(f"実行結果: {result}")
