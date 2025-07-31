from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
import time

chat = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
retriever = Chroma(persist_directory="./data", embedding_function=embeddings).as_retriever()
prompt = PromptTemplate.from_template("文章を元に質問に答えてください。\n\n文章:\n{context}\n\n質問: {input}")

# Runnable Implementation
query = "what is kazuki minemura's skill sets?"

# 文書を整形する関数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# 1. retrieve
retrieved_docs = retriever.invoke(query)
context_text = format_docs(retrieved_docs)
rag_chain = (prompt | chat | StrOutputParser())

start = time.perf_counter()
answer = rag_chain.invoke({"input": query, "context": context_text})
end = time.perf_counter()
print(f"QA in {(end - start) * 1000} milliseconds")

# 3. 出力
print("🔹 質問:")
print(query)
print("\n🔹 回答:")
print(answer)
print("\n🔹 使われたコンテキスト（文書）:")
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content[:500])  # 必要なら短くカット
    print(f"📄 Metadata: {doc.metadata}")