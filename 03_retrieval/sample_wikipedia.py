from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
  lang='ja',
)
documents = retriever.invoke(
  "大規模言語モデル"
)

print(f"検索結果: {len(documents)}件")

for document in documents:
  print("--------- 取得したデータ ---------")
  print(document.metadata)
  print("--------- 取得したテキスト ---------")
  print(document.page_content[:100])