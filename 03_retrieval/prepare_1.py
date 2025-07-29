from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("./sample.pdf")
documents = loader.load() # 一ページごとに一つの文章が作成される

print(f"ドキュメントの数: {len(documents)}")

print(f"一つ目のドキュメントの内容: {documents[0].page_content}")
print(f"一つ目のドキュメントのメタデータ: {documents[0].metadata}")

