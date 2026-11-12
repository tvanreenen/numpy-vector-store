from numpy_vector_store import VectorStore

store = VectorStore[str](dimensions=2, normalize=False)
store.add([[1.0, 0.0], [5.0, 0.0], [0.0, 1.0]], ["near", "strong", "other"])

query = [1.0, 0.0]
print("cosine:", [hit.metadata for hit in store.cosine_search(query, top_k=2)])
print("dot:", [hit.metadata for hit in store.dot_search(query, top_k=2)])
print("distance:", [hit.metadata for hit in store.euclidean_search(query, top_k=2)])
