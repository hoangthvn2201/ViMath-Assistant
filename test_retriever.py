# # Create FAISS index from a list of texts
# from app.embeddings.text_encoder import TextEncoder
# import faiss
# import numpy as np
# import pickle

# texts = ["Giải phương trình x^2 - 2x + 1 = 0", "Tính diện tích hình tròn có bán kính 5cm", ...]
# encoder = TextEncoder()
# embeddings = encoder.encode(texts)

# # Save corpus
# with open("data/processed/corpus.pkl", "wb") as f:
#     pickle.dump(texts, f)

# # Create and save FAISS index
# dim = embeddings.shape[1]
# index = faiss.IndexFlatIP(dim)
# index.add(embeddings)
# faiss.write_index(index, "data/faiss_index/math.index")


# from app.retriever import MathRetriever

# retriever = MathRetriever(
#     index_path="data/faiss_index/math.index",
#     corpus_path="data/processed/corpus.pkl",
#     encoder_model="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
# )

# query = "Tìm nghiệm của phương trình x^2 - 5x + 6 = 0"
# results = retriever.retrieve(query, top_k=3)

# for text, score in results:
#     print(f"Score: {score:.3f} | Text: {text}")



from PIL import Image
from app.embeddings.image_encoder import ImageEncoder

encoder = ImageEncoder()

img = Image.open(r"C:\Users\huyho\OneDrive\Desktop\MathRAG\image.png")
embedding = encoder.encode(img)

print("Image embedding shape:", embedding.shape)