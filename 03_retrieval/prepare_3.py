from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

loader = PyMuPDFLoader("./sample.pdf")
documents = loader.load() # 一ページごとに一つの文章が作成される

test_splitter = SpacyTextSplitter(
  chunk_size=200,
  pipeline="en_core_web_sm",
)
splitted_documents = test_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
  model="text-embedding-ada-002",
)

database = Chroma(
  persist_directory="./data",
  embedding_function=embeddings,
)

database.add_documents(splitted_documents)

print(f"データベースの作成が完了しました。")
