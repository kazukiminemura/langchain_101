from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_chroma import Chroma
import time

chat = ChatOpenAI(model="gpt-3.5-turbo")

embeddings = OpenAIEmbeddings(
  model="text-embedding-ada-002",
)

database = Chroma(
  persist_directory="./data",
  embedding_function=embeddings,
)

retriever = database.as_retriever()

prompt = PromptTemplate(
    template="""文章を元に質問に答えてください。

文章:
{context}

質問: {input}
""",
    input_variables=["context", "input"],
)

# RAG Chain
basic_qa_chain = create_stuff_documents_chain(
    llm = chat,
    prompt = prompt,
)

rag_chain = create_retrieval_chain(
  retriever = retriever,
  combine_docs_chain = basic_qa_chain,
)

query = "what is kazuki minemura's skill sets?"

start = time.perf_counter()
result = rag_chain.invoke({"input": query})
end = time.perf_counter()
print(f"QA in {(end - start) * 1000} milliseconds")

print(result["answer"])
print(result["context"]) 