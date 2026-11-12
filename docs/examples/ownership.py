from numpy_vector_store import VectorStore

payload = {"title": "Original"}
store = VectorStore[dict[str, str]](dimensions=2)
store.add([[1.0, 0.0]], [payload])

row = store.get(0)
assert row is not None
vector, metadata = row
vector[0] = 99.0  # Your own vector copy.
metadata["title"] = "Reviewed"  # The shared metadata object.

print(float(store.vectors[0, 0]))
print(payload["title"])
print(store.get(100))  # A missing row returns None.
