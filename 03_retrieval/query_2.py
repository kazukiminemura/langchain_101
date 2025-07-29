from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_chroma import Chroma
import time



embeddings = OpenAIEmbeddings(
  model="text-embedding-ada-002",
)

database = Chroma(
  persist_directory="./data",
  embedding_function=embeddings,
)

query = "what is kazuki minemura's skill sets?"

start = time.perf_counter()
documents = database.similarity_search(query)
end = time.perf_counter()
print(f"Documents retrieved in {(end - start) * 1000} milliseconds")

documents_string = ""

for document in documents:
    documents_string += f"""
    ------------------------
    {document.page_content}
    """

prompt = PromptTemplate(
    template="""文章を元に質問に答えてください。

文章:
{document}

質問: {query}
""",
    input_variables=["document", "query"],
)

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)

start = time.perf_counter()
result = chat.invoke(
    [
        HumanMessage(
            content=prompt.format(
                document=documents_string, query=query))
    ]
)
end = time.perf_counter()
print(f"Response generated in {(end - start) * 1000} milliseconds")

print(result.content)
