import pandas as pd
import pytest

from eval.benchmarks.gaia.adapter import GaiaAdapter


def test_gaia_adapter_schema_mapping(monkeypatch: pytest.MonkeyPatch):
    frame = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "question": "What is 2+2?",
                "final_answer": "4",
                "Level": "1",
                "source": "unit-test",
            },
            {
                "task_id": "t2",
                "question": "What is 3+3?",
                "final_answer": "6",
                "Level": "2",
                "source": "unit-test",
            },
        ]
    )

    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)

    adapter = GaiaAdapter()
    samples = adapter.load_samples(split="validation", limit=1, seed=123, options={"parquet_path": "dummy"})

    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "t1"
    assert sample.prompt == "What is 2+2?"
    assert sample.ground_truth == "4"
    assert sample.metadata["Level"] == "1"
    assert sample.metadata["source"] == "unit-test"


def test_gaia_adapter_maps_required_files_into_assets(monkeypatch: pytest.MonkeyPatch):
    frame = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "question": "Use the attached file.",
                "final_answer": "ok",
                "file_name": "doc.pdf",
                "file_path": "2023/validation/doc.pdf",
            }
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)

    adapter = GaiaAdapter()
    samples = adapter.load_samples(
        split="validation",
        limit=1,
        seed=0,
        options={"hf_base_uri": "hf://datasets/gaia-benchmark/GAIA"},
    )

    assert len(samples) == 1
    sample = samples[0]
    required_files = sample.assets.get("required_files")
    assert isinstance(required_files, list)
    assert required_files == [
        {
            "name": "doc.pdf",
            "dataset_path": "2023/validation/doc.pdf",
            "source_uri": "hf://datasets/gaia-benchmark/GAIA/2023/validation/doc.pdf",
        }
    ]


def test_gaia_adapter_schema_error_lists_discovered_columns(monkeypatch: pytest.MonkeyPatch):
    frame = pd.DataFrame([{"id": "x", "question": "q", "unexpected": "value"}])
    monkeypatch.setattr(pd, "read_parquet", lambda _path: frame)

    adapter = GaiaAdapter()
    with pytest.raises(ValueError) as exc:
        adapter.load_samples(split="validation", limit=None, seed=0, options={"parquet_path": "dummy"})

    message = str(exc.value)
    assert "GAIA schema error" in message
    assert "discovered columns" in message
    assert "unexpected" in message
