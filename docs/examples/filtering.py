from numpy_vector_store import VectorStore

store = VectorStore[dict[str, str]](dimensions=2)
store.add(
    [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
    [
        {"title": "Install", "product": "A"},
        {"title": "Configure", "product": "B"},
        {"title": "Deploy", "product": "A"},
    ],
)

rows = [i for i, payload in enumerate(store.metadata) if payload["product"] == "A"]
hits = store.cosine_search([1.0, 0.0], top_k=2, within_rows=rows)
print([(hit.index, hit.metadata["title"]) for hit in hits])
