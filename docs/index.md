---
title: Vector search, without the infrastructure.
description: Exact, in-process vector search powered by NumPy. Get started with a small typed Python API and your own embeddings.
---

<p class="edition">Python library · 0.7.0</p>

# Vector search, without the infrastructure.

<p class="intro">Exact, in-process vector search powered by NumPy.</p>

Keep embeddings and metadata in your Python process. Search by cosine similarity,
dot product, or Euclidean distance, then save a trusted local archive when you
need it. Bring your own embedding model and application data.

NumPy Vector Store fits small-to-medium collections that can stay in memory.
Every search scores every selected row; there is no approximate index or
background service.

## Install

```sh
uv add numpy-vector-store
```

Using pip:

```sh
python -m pip install numpy-vector-store
```

Requires **Python 3.11 or newer** and **NumPy 1.23.2 or newer**.
Python 3.11–3.14 are tested. These docs describe the published **0.7.0** API.

## Your first search

Use small vectors to see the complete workflow. Replace them with embeddings
from your model when you integrate the library.

```python
from numpy_vector_store import VectorStore

store = VectorStore[str](dimensions=3)
store.add(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ["x-axis", "y-axis", "z-axis"],
)

for hit in store.cosine_search([0.9, 0.1, 0.0], top_k=2):
    print(f"{hit.metadata}: {hit.value:.3f}")
```

```text
x-axis: 0.994
y-axis: 0.110
```

Each result is a `VectorHit` with the original row `index`, the metric `value`,
and your `metadata` payload. Vectors are stored as `float32` and normalized by
default. Choose `normalize=False` when magnitude matters.

## Keep going

Read the [API reference](reference.md) for supported methods and signatures.

The package is pre-1.0. Version 0.7 publishes the contract intended for 1.0;
it does not yet make a 1.x stability commitment.
