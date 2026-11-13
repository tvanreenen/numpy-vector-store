from pathlib import Path
from tempfile import TemporaryDirectory

from numpy_vector_store import VectorStore

with TemporaryDirectory() as directory:
    archive = Path(directory) / "catalog.npz"

    # Prepare once, then package the archive with the application.
    prepared = VectorStore[str](dimensions=2)
    prepared.add([[1.0, 0.0], [0.0, 1.0]], ["install", "deploy"])
    prepared.save(archive)

    # Each worker would perform its own open() during startup.
    replica_a = VectorStore[str].open(archive)
    replica_b = VectorStore[str].open(archive)
    query = [1.0, 0.0]
    print("replica A:", replica_a.cosine_search(query, top_k=1)[0].metadata)
    print("replica B:", replica_b.cosine_search(query, top_k=1)[0].metadata)

    # Local mutation is not synchronization or persistence.
    replica_a.clear()
    print("rows after clearing A:", len(replica_a), len(replica_b))
    print("rows in a fresh load:", len(VectorStore[str].open(archive)))
