"""Tests for the repository performance benchmark commands."""

import gc
import json

from benchmarks import benchmark


def test_measurement_keeps_cyclic_gc_outside_timed_trials():
    """Test measured operations run without changing the caller's GC state."""
    collection_states = []
    garbage_collection_enabled = gc.isenabled()
    gc.enable()
    try:
        benchmark._measure(
            lambda: collection_states.append(gc.isenabled()),
            warmup=0,
            repetitions=2,
        )

        assert collection_states == [False, False]
        assert gc.isenabled() is True
    finally:
        if not garbage_collection_enabled:
            gc.disable()


def test_ingest_benchmark_reports_workload_and_measurement(capsys):
    """Test ingestion output records its inputs and timing method."""
    exit_code = benchmark.main(
        [
            "ingest",
            "--rows",
            "4",
            "--dimensions",
            "2",
            "--batch-size",
            "3",
            "--warmup",
            "0",
            "--repetitions",
            "1",
        ]
    )

    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["schema_version"] == 1
    assert result["benchmark"] == "ingest"
    workload = result["workload"]
    vector_digest = workload.pop("prepared_vectors_sha256")
    assert len(vector_digest) == 64
    assert workload == {
        "rows": 4,
        "dimensions": 2,
        "batch_size": 3,
        "add_calls": 2,
        "normalize": True,
        "seed": 20260821,
        "random_generator": "NumPy Generator(PCG64)",
        "prepared_vector_bytes": 32,
        "timed_region": "construct VectorStore and call add() for every batch",
    }
    assert result["measurement"]["warmup_trials"] == 0
    assert result["measurement"]["measured_trials"] == 1
    assert len(result["measurement"]["samples_ns"]) == 1
    assert result["measurement"]["summary_statistic"] == "median"
    assert result["environment"]["timer"]["name"] == "perf_counter_ns"


def test_search_benchmark_reports_workload_and_per_query_measurement(capsys):
    """Test search output states what each measured trial includes."""
    exit_code = benchmark.main(
        [
            "search",
            "--rows",
            "4",
            "--dimensions",
            "2",
            "--queries",
            "2",
            "--top-k",
            "2",
            "--metric",
            "dot",
            "--no-normalize",
            "--warmup",
            "0",
            "--repetitions",
            "1",
        ]
    )

    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["benchmark"] == "search"
    workload = result["workload"]
    vector_digest = workload.pop("prepared_vectors_sha256")
    query_digest = workload.pop("prepared_queries_sha256")
    assert len(vector_digest) == 64
    assert len(query_digest) == 64
    assert workload == {
        "rows": 4,
        "dimensions": 2,
        "queries_per_trial": 2,
        "metric": "dot",
        "top_k": 2,
        "normalize": False,
        "seed": 20260821,
        "random_generator": "NumPy Generator(PCG64)",
        "stored_vector_bytes": 32,
        "timed_region": "run every query through the public unfiltered search API",
    }
    assert result["measurement"]["warmup_trials"] == 0
    assert result["measurement"]["measured_trials"] == 1
    assert result["measurement"]["median_ns_per_query"] > 0
    assert result["measurement"]["median_queries_per_second"] > 0
