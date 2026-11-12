---
title: Performance and practical limits
description: Understand exact-search costs, memory use, workload sizing, and the conditions behind the published benchmark measurements.
---

# Performance and practical limits

Exact search scores every selected row. Budget for vector dimensions, metadata,
temporary arrays, and your query rate. Validate representative workloads on your
hardware before deciding whether an in-process collection fits.

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
