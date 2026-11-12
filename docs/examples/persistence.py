from pathlib import Path
from tempfile import TemporaryDirectory

from numpy_vector_store import VectorStore

with TemporaryDirectory() as directory:
    path = Path(directory) / "vectors.npz"
    store = VectorStore[str](dimensions=2)
    store.add([[1.0, 0.0]], ["first"])
    store.save(path)  # Write and bind the destination.

    store.add([[0.0, 1.0]], ["second"])
    store.save()  # Update the bound archive explicitly.

    loaded = VectorStore[str].open(path)
    print(len(loaded), loaded.dimensions, loaded.normalize)

    loaded.clear()  # This changes memory only.
    loaded.reload()  # Discard memory and reread the archive.
    print([hit.metadata for hit in loaded.cosine_search([0.0, 1.0])])
