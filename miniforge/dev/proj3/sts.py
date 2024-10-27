# step 1

from sentence_transformers import SentenceTransformer


# step2
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# step3
sentences1 = "집에 갑시다."
sentences2 = "안녕하세요"


# step4
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

print(embeddings1.shape)


# step5
similarities = model.similarity(embeddings1, embeddings2)
print(similarities)
