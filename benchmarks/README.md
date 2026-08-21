# Performance benchmarks

The benchmark command measures the public ingestion and exact-search APIs with
seeded `float32` inputs. It writes one JSON object to standard output so the
complete record can be kept with an investigation or release.

## Search

```bash
uv run python benchmarks/benchmark.py search \
  --rows 10000 \
  --dimensions 384 \
  --queries 20 \
  --metric cosine \
  --top-k 10 \
  --warmup 2 \
  --repetitions 7 \
  > /tmp/nvs-search.json
```

The store and query vectors are generated before timing. Store construction,
ingestion, and random input generation are therefore excluded. One trial runs
every prepared query through the selected public, unfiltered search method and
includes metric calculation, partial top-k selection, deterministic result
ordering, and `VectorHit` construction.

`--metric` accepts `cosine`, `dot`, or `euclidean`. The store normalizes vectors
by default; pass `--no-normalize` to exercise raw-vector behavior.

## Ingestion

```bash
uv run python benchmarks/benchmark.py ingest \
  --rows 10000 \
  --dimensions 384 \
  --batch-size 1 \
  --warmup 2 \
  --repetitions 7 \
  > /tmp/nvs-ingest.json
```

Vectors and metadata are generated before timing. Each trial constructs a new
store and fills it through the public `add()` method using the requested batch
size. The duration includes validation, normalization, row-capacity growth,
metadata handling, and every `add()` call. It excludes input generation and
disposes of the completed store after the timer stops.

Use `--batch-size 1` to exercise repeated small additions. Use a larger batch
size to model applications that already receive embeddings in batches. This is
an application-level choice: the store still preserves insertion order and the
same row semantics in either case.

## Measurement record

Both commands use the same measurement rules:

- An explicit NumPy `Generator(PCG64)` produces seeded inputs. The record keeps
  the NumPy version and SHA-256 digests of the prepared arrays because NumPy's
  distribution methods do not promise identical streams across every version.
- `--warmup` complete trials run first and are discarded.
- Garbage collection runs before, not during, each measured trial.
- `time.perf_counter_ns()` records each complete trial.
- The JSON retains every nanosecond sample and reports their median.
- Derived per-query or rows-per-second values come from that median.

The environment record includes the package version, Git revision and dirty
state when available, Python and NumPy versions, platform and processor data,
timer resolution, common numerical-library thread variables, and
`numpy.show_config()` output. NumPy's configuration identifies build details
such as the BLAS implementation. The command does not force a thread count; set
the relevant environment variable before starting Python when a comparison
requires fixed threading.

Run either subcommand with `--help` for its complete option list. All row,
dimension, batch, query, top-k, and repetition counts must be positive; warmup
may be zero.

## Interpreting results

The generated vectors and query sequence are repeatable in the recorded NumPy
environment, and their digests make exact input equality checkable. Elapsed
time is still specific to the recorded machine and software environment.
Compare equivalent workloads and inspect the raw samples for variance. A
single run from another computer is useful evidence for that computer, not a
portable latency promise.

Shared CI enforces structural properties instead of time limits: repeated
additions reuse geometric row capacity, small top-k searches use partial
selection before sorting, and unfiltered searches operate on the stored vector
matrix without first copying it. This avoids noisy failures caused by variable
runner load while keeping the intended complexity visible in tests.
