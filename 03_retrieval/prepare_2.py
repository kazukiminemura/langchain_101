from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter

loader = PyMuPDFLoader("./sample.pdf")
documents = loader.load() # 一ページごとに一つの文章が作成される

test_splitter = SpacyTextSplitter(
  chunk_size=300,
  pipeline="en_core_web_sm",
)
splitted_documents = test_splitter.split_documents(documents)

print(f"ドキュメントの数: {len(documents)}")
print(f"分割後のドキュメントの数: {len(splitted_documents)}")
