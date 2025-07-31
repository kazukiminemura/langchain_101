from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever
from langchain.retrievers.re_phraser import RePhraseQueryRetriever
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

retriever = WikipediaRetriever(
  lang='ja',
  doc_content_chars_max=500,
)

llm = ChatOpenAI(
  temperature=0
)

prompt = PromptTemplate(
  input_variables=["question"],
  template="""以下の質問からWikipediaで検索するべきキーワードを抽出してください。
  質問:{question}
  """
)

llm_chain = prompt | llm | StrOutputParser() # RunnableSequenceとして自動構築される

re_phrase_query_retriever = RePhraseQueryRetriever(
  llm_chain = llm_chain,
  retriever = retriever,
)

documents = re_phrase_query_retriever.invoke("私はラーメンが好きです。ところでバーボンウィスキーとは何ですか？")

print(documents)