from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever
from langchain.prompts import PromptTemplate

chat = ChatOpenAI()

retriever = WikipediaRetriever(
  lang='ja',
  doc_content_chars_max=500,
  top_k_results=2,
)

prompt = PromptTemplate(
 template="""文章を元に質問に答えてください

文章:
{context}

質問: {input}
""",
input_variables=["context", "intput"]
)

basic_qa_chain = create_stuff_documents_chain(
  llm=chat,
  prompt=prompt
)

chain = create_retrieval_chain(
  retriever = retriever,
  combine_docs_chain=basic_qa_chain,
)

result = chain.invoke({"input": "バーボンウィスキーとは？"})

source_documents = result["context"]

print(f"検索結果: {len(source_documents)}件")

for document in source_documents:
  print("--------- 取得したデータ ---------")
  print(document.metadata)
  print("--------- 取得したテキスト ---------")
  print(document.page_content[:100])
print("--------------返答----------------")
print(result["answer"])
