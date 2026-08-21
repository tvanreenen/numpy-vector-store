# Persistence compatibility fixtures

## `vector-store-0.4.0-format-v1.npz`

This archive was written by the published `numpy-vector-store==0.4.0` wheel on
Python 3.11 with NumPy 1.23.2, the oldest runtime supported by 0.4 and later.
Its SHA-256 digest is
`570f837ac7652a860fb2ee41abe36a0e028318903ea55e0eeee3daa269119894`.

The store uses three dimensions with `normalize=False` and contains two rows of
built-in metadata. Keeping the generated binary in the repository proves that a
current release can open an actual format-version-1 archive from 0.4; generating
an equivalent archive with the current source would not test that promise.

The fixture does not restore support for unversioned archives, guarantee that
0.4 can open files written by future formats, or make arbitrary pickled metadata
portable when its application classes or dependencies are unavailable.

The archive was generated in an isolated environment with:

```python
from pathlib import Path

from numpy_vector_store import VectorStore

output = Path("vector-store-0.4.0-format-v1.npz")
store = VectorStore[dict[str, object]](dimensions=3, normalize=False)
store.add(
    [[1.5, -2.0, 0.25], [0.0, 3.0, 4.0]],
    [
        {"id": "alpha", "tags": ("legacy", 4)},
        {"id": "beta", "active": True},
    ],
)
store.save(output)
```
