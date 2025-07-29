from xml.dom.minidom import Document
from langchain_openai import OpenAIEmbeddings
from numpy import dot
from numpy.linalg import norm

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
)

query_vector = embeddings.embed_query("飛行機の最高速度はどのくらいですか？")

print(f"ベクトル化されたクエリ: {query_vector[:5]}") # 1536次元のベクトルの最初の5つの要素を表示

document_1_vector = embeddings.embed_query("飛行機の最高速度は約900キロメートル毎時です。")
document_2_vector = embeddings.embed_query("鶏肉を適切に下味をつけた後、中火で焼きながらたまに裏返し、約20分焼きます。")

cos_sim_1 = dot(query_vector, document_1_vector) / (norm(query_vector) * norm(document_1_vector))
print(f"クエリとドキュメント1のコサイン類似度: {cos_sim_1}")

cos_sim_2 = dot(query_vector, document_2_vector) / (norm(query_vector) * norm(document_2_vector))
print(f"クエリとドキュメント2のコサイン類似度: {cos_sim_2}")
