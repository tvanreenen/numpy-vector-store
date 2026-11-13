---
title: Shipping a store with your application
description: Build a trusted vector archive, include it in an application image, and serve local searches from independent workers and cloud replicas.
---

# Shipping a store with your application

You can deploy multiple workers or cloud replicas without operating a shared
vector database. Prepare a trusted `.npz` archive, include it in your application
image, and let each worker load its own in-memory store. Searches stay local;
your build and deployment process distributes the collection.

This fits read-mostly collections that can be refreshed through an application
rollout and fit comfortably in each worker's memory. Examples include a packaged
documentation corpus, a reference catalog, or a collection of known examples.
The collection can be small even when the application has many replicas.

## Prepare the archive before deployment

Generate embeddings in a trusted preparation step, add the vectors and aligned
metadata to a store, and save the archive. Include that file in the application
image with the code and dependencies that will read it. The preparation step
does not need to run again in each serving replica.

Use the same embedding model and preprocessing for query vectors as for the
stored collection. The archive records dimensions and normalization, but does
not identify the embedding model. Keep the model identity and collection
revision in your application's build metadata or configuration.

!!! warning "Package trusted data"
    Archives contain pickle-backed metadata. Opening an untrusted archive can
    execute code. Treat the archive as a trusted build input, not as a safe user
    upload. Custom metadata classes must remain importable under compatible
    module paths in the serving application. Simple payloads such as strings,
    dictionaries, and record identifiers can reduce those dependencies.

## Load once in each worker

During each worker's startup, call `VectorStore.open()` with an explicit path
to the packaged archive and retain the returned store for requests. Avoid
reopening it on every search. Finish loading and validate a representative
query before marking the worker ready to serve.

Opening loads the collection into memory and closes the archive; it is not a
memory-mapped or disk-backed search. Query serving needs no write access to the
packaged file. Keep the serving store unchanged for concurrent reads, including
its shared metadata payloads. If it can mutate, follow the application-level
synchronization rules in [state, views, and threading](ownership.md).

The runnable example below uses a temporary directory in place of an image's
data directory. It builds one archive and opens it twice in one process to
illustrate independent loads. In a deployment, each worker performs its own
`open()` at startup:

```python
--8<-- "docs/examples/deployment.py"
```

```text
replica A: install
replica B: install
rows after clearing A: 0 2
rows in a fresh load: 2
```

Clearing the first store changes only that instance's memory. It neither changes
the other instance nor writes the packaged archive. Processes and cloud
replicas likewise have no automatic store-update channel between them.

## Budget memory per worker

Plan for each worker to hold a separate loaded collection. Four workers in one
container need memory for four stores, plus the application and request work.
Do not size a container from the compressed archive's file size alone.

The vector matrix needs `rows × dimensions × 4` bytes before metadata, spare
capacity, and temporary arrays. Loading and validating the archive creates
temporary arrays too. Measure startup peak memory, steady-state memory, and
representative concurrent searches with your actual metadata and deployment
limits. See [performance and practical limits](performance.md) for metric and
filtering costs.

## Update through your rollout

Build a new archive and application image when the collection changes. Deploy
that revision through the same process you use for code, and keep the matching
query model and metadata dependencies with it. Retaining the previous image
also retains its packaged collection for rollback.

A rolling deployment can temporarily serve both old and new collections. If
that is unacceptable, your application or deployment system must coordinate the
cutover; the store does not provide cross-replica consistency. Track the
collection revision separately from the library's package version.

`reload()` rereads a bound local path; it does not fetch a new image or notify
other workers. `save()` writes explicitly and does not distribute changes.
Avoid saving to the packaged archive in serving workers. If you introduce a
shared writable destination, you also need external writer coordination;
atomic file replacement alone is not a multi-writer protocol.

## When to add a shared service

Independent replicas are useful when distributing a collection with each image
is simpler than operating another service. Reconsider the design when each
worker can no longer afford its memory or search cost, or when frequent writes
must become visible across replicas without an application rollout. Those needs
call for additional application coordination or a shared, possibly indexed
data service, rather than an implicit promise from a local in-memory store.
