# Eval Framework (GAIA-first)

This folder contains a transparent, extensible LLM-agent evaluation framework.

Primary goal:
- Run reproducible benchmark evaluations (GAIA first) using the real `agent1` runtime.
- Persist auditable per-sample traces and aggregate metrics.
- Keep benchmark integration open via adapters.

## What is in this folder

- `score.py`
  GAIA scorer reference logic (normalization + exact match behavior).
- `test.ipynb`
  Notebook used for quick GAIA parquet inspection/loading.
- `test.py`
  Convenience entrypoint that runs `eval.cli main(["run"])`.
- `cli.py`
  CLI interface: `run`, `summarize`, and `report`.
- `report_html.py`
  Standalone HTML report bundle renderer used by the CLI report flow. It writes
  a main summary page plus per-sample detail pages.
- `requirements.txt`
  Eval-specific dependency list.

### Core runtime modules

- `core/contracts.py`
  Public dataclasses/protocols:
  - `EvalConfig`
  - `BenchmarkSample`
  - `ScoreResult`
  - `ExecutionResult`
  - `SampleResult`
  - `RunSummary`
  - `BenchmarkAdapter` protocol
  - `AgentExecutor` protocol
  - `ArtifactWriter` protocol

- `core/runner.py`
  Orchestration engine:
  - `run_evaluation(config)`
  - `run_evaluation_async(config)`
  - `summarize_run(run_dir)`

- `core/artifacts.py`
  Artifact writer for:
  - `manifest.json`
  - `samples.jsonl`
  - `events.jsonl`
  - `errors.jsonl`
  - `summary.json`
  - `summary.md`

- `core/redaction.py`
  Secret/token redaction before artifact writes.

### Benchmark modules

- `benchmarks/registry.py`
  Adapter registration and lookup (`register_adapter`, `get_adapter`, `list_adapters`).

- `benchmarks/gaia/adapter.py`
  GAIA adapter:
  - loads parquet split(s)
  - maps schema columns
  - builds prompt
  - scores prediction
  - aggregates totals + per-level slices

- `benchmarks/gaia/scorer.py`
  Wrapper around `score.py::question_scorer` that returns `ScoreResult` with diagnostics.

### Executors

- `executors/agent1_runtime_executor.py`
  Main executor. Runs each sample through `agent_build/agent1/runtime.py` with isolation and trace capture.

- `executors/echo_executor.py`
  Minimal deterministic executor for smoke tests.

### Configs

- `configs/gaia_validation.yaml`
  Default run config (validation-first).

## GAIA dataset usage

Current GAIA split mapping:
- `validation` -> `2023/validation/metadata.parquet`
- `test` -> `2023/test/metadata.parquet`

Default dataset base URI:
- `hf://datasets/gaia-benchmark/GAIA/`

Equivalent quick load example:

```python
import pandas as pd

splits = {
    "test": "2023/test/metadata.parquet",
    "validation": "2023/validation/metadata.parquet",
}
df = pd.read_parquet("hf://datasets/gaia-benchmark/GAIA/" + splits["test"])
```

### GAIA schema mapping rules

Prompt column candidates (first match wins):
- `question`, `Question`

Ground-truth column candidates:
- `final_answer`, `Final answer`, `answer`

Sample-id column candidates:
- `task_id`, `id`
- fallback: row index

Metadata:
- all non-prompt, non-ground-truth, non-id columns are retained in `sample.metadata`.

If required columns are missing:
- adapter raises `ValueError` with discovered column list.

## Scoring behavior

GAIA scoring path uses `score.py::question_scorer`.

High-level behavior:
- numeric GT: normalize symbols (`$`, `%`, `,`) then numeric compare
- list GT (`,` or `;`): split and compare element-wise
- string GT: normalized exact compare (lowercase/whitespace and punctuation handling)

Scorer wrapper returns:
- `correct` (bool)
- `score` (`1.0` or `0.0`)
- `reason` (`exact_match` or `mismatch`)
- `normalized_prediction`
- `normalized_ground_truth`

## Running evaluations

## 1. Install dependencies

```bash
pip install -r eval/requirements.txt
```

If you use tests:

```bash
pip install pytest
```

## 2. Run default GAIA validation

```bash
python -m eval.cli run --config eval/configs/gaia_validation.yaml
```

By default, `run` also generates `report.html` inside the run directory.
That summary page links to per-sample detail pages under a sibling
`report_samples/` directory.
Use `--no-report` to skip that post-processing step.

## 3. Override common options from CLI

```bash
python -m eval.cli run \
  --config eval/configs/gaia_validation.yaml \
  --split validation \
  --limit 50 \
  --executor agent1 \
  --timeout-s 120 \
  --benchmark-option shuffle=true \
  --executor-option enable_memory=true
```

Eval runs default to `enable_memory=true` and `enable_auto_title=false`, so
per-sample memory artifacts are preserved while auto-title background work stays
disabled during benchmarking.

To generate and persist model-assisted agent step summaries for the per-sample report:

```bash
python -m eval.cli run \
  --config eval/configs/gaia_validation.yaml \
  --summarize-agent-steps \
  --summary-provider gemini \
  --summary-model gemini-3-pro-preview
```

To add optional PromptFunction summaries for ambiguous non-correct samples:

```bash
python -m eval.cli run \
  --config eval/configs/gaia_validation.yaml \
  --summarize-failure-causes \
  --summary-provider gemini \
  --summary-model gemini-3-pro-preview
```

Deterministic run analysis now persists automatically after each run, even when
`--no-report` is used. It adds:
- `result.failure_analysis` to non-correct samples in `samples.jsonl`
- `metrics.analysis` to `summary.json`
- correlation, cohort, and failure-cause aggregates to `summary.md`

`--summary-provider` and `--summary-model` are shared by both agent-step
summaries and failure-cause summaries, so one provider/model pair configures all
PromptFunction-based eval summarization.

To skip the automatic HTML report:

```bash
python -m eval.cli run \
  --config eval/configs/gaia_validation.yaml \
  --no-report
```

## 4. Summarize an existing run directory

```bash
python -m eval.cli summarize --run-dir eval/runs/<timestamp_runid>
```

This rewrites `summary.json` and `summary.md`, and also regenerates the default
HTML report bundle for that run:
- `report.html`
- `report_samples/<index>_<sample_id>.html`

Pass `--summarize-agent-steps` to regenerate persisted agent-step summaries in
`samples.jsonl` before rebuilding the report.

Pass `--summarize-failure-causes` to regenerate optional PromptFunction-based
failure summaries for low-confidence non-correct samples before rebuilding the
report.

## 5. Generate or regenerate an HTML report manually

```bash
python -m eval.cli report \
  --run-dir eval/runs/<timestamp_runid> \
  --output report.html
```

If `--output` is omitted, the report command writes `<run-id>_report.html` in the current directory.

The report command now generates a static multipage bundle:
- the main summary page at the exact `--output` path
- a sibling sample-detail directory named `<output_stem>_samples/`

Each sample row in the summary page links to a dedicated detail page containing:
- prompt, ground truth, prediction, and score diagnostics
- deterministic failure analysis for non-correct samples, plus optional AI summaries
- GAIA sample metadata plus side-by-side annotator vs. agent execution comparison
- usage snapshots, warnings, and staged attachments when present
- matched error records and a curated event timeline
- raw event JSON behind a collapsible `<details>` block

Use `--summarize-agent-steps` on `report` or `summarize` to generate and persist
PromptFunction-based agent step summaries before the HTML bundle is built.

Use `--summarize-failure-causes` on `run`, `summarize`, or `report` to generate
and persist optional PromptFunction-based summaries for ambiguous failures.

The summary page now leads with benchmark-agnostic analysis:
- correlation overview for correctness, runtime errors, tokens, latency, calls, and distinct tools
- cohort tables comparing correct, incorrect, and runtime-error samples
- aggregated failure-cause counts
- an optional secondary level breakdown when the benchmark exposes usable level metadata

## 6. Alternate entrypoint

```bash
python eval/test.py
```

## Python API usage

```python
from eval.core.contracts import EvalConfig
from eval.core.runner import run_evaluation

config = EvalConfig(
    benchmark="gaia",
    split="validation",
    limit=20,
    executor="agent1",
    output_root="eval/runs",
    continue_on_error=True,
)

summary = run_evaluation(config)
print(summary.run_dir, summary.accuracy)
```

Important:
- `run_evaluation()` cannot run inside an already-running event loop.
- In async contexts use `run_evaluation_async()`.

## Artifact contract

Each run writes to:
- `eval/runs/<YYYYMMDDTHHMMSSZ>_<run_id>/`

Required files:
- `manifest.json`
  - run config snapshot
  - git SHA (if available)
  - filtered environment snapshot
  - start/end/duration
  - artifact paths
- `samples.jsonl`
  - one line per sample
  - prompt, ground truth, metadata
  - `SampleResult` payload (status, prediction, score, usage, error)
- `events.jsonl`
  - flattened event stream entries
  - includes `sample_id` and `event_index`
- `errors.jsonl`
  - only failed samples
- `summary.json`
  - final aggregate metrics and paths
- `summary.md`
  - human-readable summary
- `report.html`
  - generated automatically by `python -m eval.cli run` unless `--no-report` is set
  - main summary page with run metrics and a linked sample table
- `report_samples/`
  - generated alongside `report.html`
  - one static HTML page per sample, named `<index>_<sample_id>.html`

## Redaction/safety in artifacts

All written payloads are redacted by default patterns plus optional custom regex.

Default token-like redactions include:
- `sk-...`
- `hf_...`
- `jina_...`
- `pplx-...`
- `AIza...`
- `Bearer ...`

Keys containing sensitive terms are also redacted:
- `key`, `token`, `secret`, `password`, `authorization`, `api_key`

Add custom patterns via:
- `EvalConfig.redact_patterns`
- CLI `--redact-pattern` (repeatable)

## Agent1 executor details

`executors/agent1_runtime_executor.py` executes each sample by:
1. building a per-sample isolated memory directory under `run_dir/_memory/...`
2. creating a runtime session with `RuntimeOverrides`
3. running one user turn (`run_user_turn`) with timeout
4. capturing events via `agent.subscribe(...)`
5. computing usage delta (`before` vs `after` usage summary)
6. always shutting down session

Default executor behavior:
- timeout honored per sample
- continue on error (unless fail-fast)
- per-sample isolation by default
- non-essential eval tools are excluded by default: `memory_search`, `gmail_fetch`, `pdf_merge`, `agentmail_send`

### Runtime overrides supported (through executor options)

Forwarded to `agent_build.agent1.runtime.RuntimeOverrides`:
- `long_memory_base_path` (str)
- `enable_memory` (bool)
- `enable_auto_title` (bool)
- `include_tools` (list[str] or comma string)
- `exclude_tools` (list[str] or comma string)
- `model_override` (dict)
- `bash_prompt_approval` (bool)
- `log_level` (str)
- `enable_event_logger` (bool, default `true`)

Example:

```bash
python -m eval.cli run \
  --config eval/configs/gaia_validation.yaml \
  --executor-option include_tools='["websnapshot","bash_exec"]' \
  --executor-option exclude_tools='["gmail_fetch","pdf_merge","memory_search"]' \
  --executor-option model_override='{"provider":"openai","id":"gpt-5-mini","api":null}'
```

If you explicitly include one of the default-excluded tools, the executor will not auto-exclude it for that run.

## Config reference (`EvalConfig`)

- `benchmark`: adapter key (`gaia`)
- `split`: split name (`validation` default)
- `limit`: optional sample cap
- `seed`: optional; used when adapter supports shuffling. If omitted, shuffled order is non-deterministic.
- `timeout_s`: per-sample timeout
- `output_root`: run directory root
- `run_id`: optional explicit run id
- `benchmark_options`: adapter-specific options
- `executor`: `agent1` or `echo`
- `executor_options`: executor-specific options
- `redact_patterns`: extra regex redactions
- `continue_on_error`: keep running after sample failure
- `concurrency`: currently forced to `1` by runner (deterministic v1)

## Extending to another benchmark

1. Create adapter class implementing `BenchmarkAdapter`:
- `load_samples(...)`
- `build_prompt(sample)`
- `score(prediction, ground_truth, sample)`
- `aggregate(sample_results)`

2. Register it in `benchmarks/registry.py` using `register_adapter("name", Factory)`.

3. Run with:

```bash
python -m eval.cli run --benchmark <name>
```

No runner changes should be required if your adapter conforms to the protocol.

## Testing

Current eval tests:
- `tests/test_eval_gaia_adapter.py`
  - schema mapping and schema error behavior
- `tests/test_eval_gaia_scorer.py`
  - numeric, string, and list scoring behavior
- `tests/test_eval_runner.py`
  - continue-on-error and artifact completeness
- `tests/test_eval_executor_agent1.py`
  - per-sample memory isolation path behavior
- `tests/test_eval_redaction.py`
  - redaction of key/token-like values
- `tests/test_eval_cli.py`
  - CLI run + summarize smoke
- `tests/test_eval_runtime_overrides.py`
  - runtime override compatibility and filtering

Run:

```bash
python -m pytest -q tests/test_eval_*.py
```

## Known limitations (current v1)

- concurrency is intentionally restricted to 1 in runner
- no built-in retry policy for failed samples
- GAIA adapter assumes metadata parquet shape and candidate column names listed above
- `agent1` executor runs one prompt per sample; no multi-turn benchmark orchestration yet
- if parquet dependencies (`pyarrow`/`fastparquet`) are missing, loading GAIA fails

## Troubleshooting

### `Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'`
Install parquet support:

```bash
pip install pyarrow
```

### `PyYAML is required to read YAML configs`
Install yaml dependency:

```bash
pip install PyYAML
```

### `Unknown benchmark adapter`
List registered adapters in code (`benchmarks/registry.py`) and register your adapter.

### `Unknown executor`
Supported executors today:
- `agent1`
- `echo`

### Network/auth errors while loading GAIA from HF
Ensure Hugging Face access is configured and network allows `hf://` reads.

## Reproducibility recommendations

For strict benchmarking:
- keep `concurrency=1` (default)
- set `split`, `limit`, and `seed` explicitly
- pin model via `executor_options.model_override`
- disable non-essential tools with `include_tools`/`exclude_tools`
- disable interactive approval paths (already defaulted for eval in executor)
- archive run directory and config together
