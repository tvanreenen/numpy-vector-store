---
title: Performance and practical limits
description: Understand exact-search costs, memory use, workload sizing, and the conditions behind the published benchmark measurements.
---

# Performance and practical limits

Exact search scores every selected row. Budget for vector dimensions, metadata,
temporary arrays, and your query rate. Validate representative workloads on your
hardware before deciding whether an in-process collection fits.

## Collections and replicas

Small-to-medium describes the collection in each process, not a limit on the
number of workers or replicas. A read-mostly archive can be distributed with
an application image and loaded independently by each worker. See the
[deployment guide](deployment.md) for that pattern.

Budget for a separate in-memory store per worker, even when several workers run
inside one container. A compressed `.npz` file's size is not its loaded memory
requirement. Loading and validating an archive also allocates temporary arrays;
measure startup peak memory as well as steady-state search memory.

Consider an indexed or service-backed system when collection size, query load,
or per-worker memory makes local exact search impractical. Frequent updates that
must become visible across all replicas need application-level distribution and
coordination or a shared service; the library does not synchronize stores.

## Memory and workload shape

The active vector matrix alone needs `rows × dimensions × 4` bytes. This excludes
private spare capacity, metadata objects, temporary arrays, and Python overhead.
Filtered searches allocate a gathered vector matrix, and raw Euclidean search
uses a `float64` difference matrix. Concurrent searches multiply temporary memory.

Small top-k queries use partial selection before sorting the selected hits;
reducing `top_k` does not remove the full candidate metric calculation. Repeated
small additions reuse geometric capacity. Passing a batch still reduces Python
and validation overhead when inputs are already available together.

## Published reference measurements

The following section is shared with the standalone package README. Its commit,
machine, and dependency versions describe the measurement, not a new benchmark
run of every later release.

--8<-- "README.md:performance"
