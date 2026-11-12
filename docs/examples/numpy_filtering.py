import numpy as np

from numpy_vector_store import VectorStore

records = np.array(
    [("intro", "A", 2024), ("setup", "A", 2023), ("guide", "B", 2024)],
    dtype=[("title", "U20"), ("product", "U10"), ("year", "i4")],
)
store = VectorStore[int](dimensions=2)
store.add([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], list(range(len(records))))

mask = (records["product"] == "A") & (records["year"] >= 2024)
rows = np.flatnonzero(mask)
for hit in store.cosine_search([1.0, 0.0], within_rows=rows):
    print(hit.index, records[hit.metadata]["title"])
