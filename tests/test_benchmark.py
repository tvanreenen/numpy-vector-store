"""Tests for the repository performance benchmark commands."""

import json

from benchmarks import benchmark


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
    assert result["workload"] == {
        "rows": 4,
        "dimensions": 2,
        "batch_size": 3,
        "add_calls": 2,
        "normalize": True,
        "seed": 20260821,
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
    assert result["workload"] == {
        "rows": 4,
        "dimensions": 2,
        "queries_per_trial": 2,
        "metric": "dot",
        "top_k": 2,
        "normalize": False,
        "seed": 20260821,
        "stored_vector_bytes": 32,
        "timed_region": "run every query through the public unfiltered search API",
    }
    assert result["measurement"]["warmup_trials"] == 0
    assert result["measurement"]["measured_trials"] == 1
    assert result["measurement"]["median_ns_per_query"] > 0
    assert result["measurement"]["median_queries_per_second"] > 0
