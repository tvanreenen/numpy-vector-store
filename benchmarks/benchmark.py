"""Run reproducible ingestion and exact-search benchmarks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

from numpy_vector_store import VectorStore, __version__

_DEFAULT_SEED = 20260821
_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rows", type=_positive_int, default=10_000)
    parser.add_argument("--dimensions", type=_positive_int, default=384)
    parser.add_argument("--warmup", type=_non_negative_int, default=2)
    parser.add_argument("--repetitions", type=_positive_int, default=7)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="normalize vectors on ingestion (default: true)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark public NumPy Vector Store operations and emit one JSON record."
        )
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    ingest = subparsers.add_parser(
        "ingest",
        help="measure filling a new store from deterministic prepared inputs",
    )
    _add_common_arguments(ingest)
    ingest.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="rows supplied to each add() call (default: 1)",
    )

    search = subparsers.add_parser(
        "search",
        help="measure unfiltered exact searches over a prepared store",
    )
    _add_common_arguments(search)
    search.add_argument(
        "--metric",
        choices=("cosine", "dot", "euclidean"),
        default="cosine",
    )
    search.add_argument("--queries", type=_positive_int, default=20)
    search.add_argument("--top-k", type=_positive_int, default=10)

    return parser


def _git_state() -> dict[str, str | bool | None]:
    repository = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": dirty}


def _numpy_configuration() -> list[str]:
    output = io.StringIO()
    with warnings.catch_warnings(), redirect_stdout(output):
        warnings.filterwarnings(
            "ignore",
            message="Install `pyyaml` for better output",
            category=UserWarning,
            module=r"numpy\.__config__",
        )
        np.show_config()
    return [line.rstrip() for line in output.getvalue().splitlines() if line.strip()]


def _environment() -> dict[str, Any]:
    clock = time.get_clock_info("perf_counter")
    return {
        "package_version": __version__,
        "git": _git_state(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "timer": {
            "name": "perf_counter_ns",
            "implementation": clock.implementation,
            "resolution_seconds": clock.resolution,
        },
        "thread_environment": {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_VARIABLES
        },
        "numpy_configuration": _numpy_configuration(),
    }


def _measure(
    operation: Callable[[], object],
    *,
    warmup: int,
    repetitions: int,
) -> list[int]:
    for _ in range(warmup):
        result = operation()
        del result
        gc.collect()

    samples = []
    for _ in range(repetitions):
        gc.collect()
        garbage_collection_enabled = gc.isenabled()
        if garbage_collection_enabled:
            gc.disable()
        try:
            start = time.perf_counter_ns()
            result = operation()
            elapsed = time.perf_counter_ns() - start
        finally:
            if garbage_collection_enabled:
                gc.enable()
        samples.append(elapsed)
        del result
    return samples


def _measurement(samples_ns: list[int], *, warmup: int) -> dict[str, Any]:
    median_ns = statistics.median(samples_ns)
    return {
        "warmup_trials": warmup,
        "measured_trials": len(samples_ns),
        "summary_statistic": "median",
        "samples_ns": samples_ns,
        "median_ns": median_ns,
        "median_seconds": median_ns / 1_000_000_000,
    }


def _prepared_inputs(
    *, rows: int, dimensions: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(seed))
    vectors = generator.standard_normal(size=(rows, dimensions), dtype=np.float32)
    metadata = np.arange(rows)
    return vectors, metadata


def _array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.data).hexdigest()


def _run_ingest(arguments: argparse.Namespace) -> dict[str, Any]:
    vectors, metadata = _prepared_inputs(
        rows=arguments.rows,
        dimensions=arguments.dimensions,
        seed=arguments.seed,
    )

    def ingest() -> VectorStore[np.integer[Any]]:
        store = VectorStore[np.integer[Any]](
            dimensions=arguments.dimensions,
            normalize=arguments.normalize,
        )
        for start in range(0, arguments.rows, arguments.batch_size):
            end = min(start + arguments.batch_size, arguments.rows)
            store.add(vectors[start:end], metadata[start:end])
        return store

    samples = _measure(
        ingest,
        warmup=arguments.warmup,
        repetitions=arguments.repetitions,
    )
    measurement = _measurement(samples, warmup=arguments.warmup)
    measurement["median_rows_per_second"] = (
        arguments.rows / measurement["median_seconds"]
    )
    return {
        "schema_version": 1,
        "benchmark": "ingest",
        "environment": _environment(),
        "workload": {
            "rows": arguments.rows,
            "dimensions": arguments.dimensions,
            "batch_size": arguments.batch_size,
            "add_calls": (arguments.rows + arguments.batch_size - 1)
            // arguments.batch_size,
            "normalize": arguments.normalize,
            "seed": arguments.seed,
            "random_generator": "NumPy Generator(PCG64)",
            "prepared_vector_bytes": vectors.nbytes,
            "prepared_vectors_sha256": _array_digest(vectors),
            "timed_region": "construct VectorStore and call add() for every batch",
        },
        "measurement": measurement,
    }


def _run_search(arguments: argparse.Namespace) -> dict[str, Any]:
    vectors, metadata = _prepared_inputs(
        rows=arguments.rows,
        dimensions=arguments.dimensions,
        seed=arguments.seed,
    )
    query_generator = np.random.Generator(np.random.PCG64(arguments.seed + 1))
    queries = query_generator.standard_normal(
        size=(arguments.queries, arguments.dimensions), dtype=np.float32
    )
    store = VectorStore[np.integer[Any]](
        dimensions=arguments.dimensions,
        normalize=arguments.normalize,
    )
    store.add(vectors, metadata)

    def search() -> object:
        hits: object = None
        for query in queries:
            if arguments.metric == "cosine":
                hits = store.cosine_search(query, top_k=arguments.top_k)
            elif arguments.metric == "dot":
                hits = store.dot_search(query, top_k=arguments.top_k)
            else:
                hits = store.euclidean_search(query, top_k=arguments.top_k)
        return hits

    samples = _measure(
        search,
        warmup=arguments.warmup,
        repetitions=arguments.repetitions,
    )
    measurement = _measurement(samples, warmup=arguments.warmup)
    measurement["median_ns_per_query"] = measurement["median_ns"] / arguments.queries
    measurement["median_queries_per_second"] = (
        arguments.queries / measurement["median_seconds"]
    )
    return {
        "schema_version": 1,
        "benchmark": "search",
        "environment": _environment(),
        "workload": {
            "rows": arguments.rows,
            "dimensions": arguments.dimensions,
            "queries_per_trial": arguments.queries,
            "metric": arguments.metric,
            "top_k": arguments.top_k,
            "normalize": arguments.normalize,
            "seed": arguments.seed,
            "random_generator": "NumPy Generator(PCG64)",
            "stored_vector_bytes": store.vectors.nbytes,
            "prepared_vectors_sha256": _array_digest(vectors),
            "prepared_queries_sha256": _array_digest(queries),
            "timed_region": "run every query through the public unfiltered search API",
        },
        "measurement": measurement,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected benchmark and write its complete JSON record."""
    arguments = _build_parser().parse_args(argv)
    if arguments.benchmark == "ingest":
        result = _run_ingest(arguments)
    else:
        result = _run_search(arguments)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
