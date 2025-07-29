from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(
  model="text-embedding-ada-002",
)

database = Chroma(
  persist_directory="./data",
  embedding_function=embeddings,
)

documents = database.similarity_search("what is kazuki minemura's skill sets?")
print(f"ドキュメントの数: {len(documents)}")

for document in documents:
    print(f"ドキュメントの内容: {document.page_content}")
    print("-----")