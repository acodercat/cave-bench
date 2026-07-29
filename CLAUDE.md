# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

cave-bench evaluates [CaveAgent](https://github.com/acodercat/cave-agent) — an agent that calls tools by **writing and executing Python code** — and compares it against traditional **JSON function calling** (via LiteLLM). Both agent styles run through one shared evaluation pipeline so their metrics are directly comparable.

## Commands

Uses `uv` (Python >= 3.12). Prefix Python invocations with `uv run`.

```bash
uv sync                                       # install deps into .venv
cp models.toml.example models.toml            # then fill in API keys (models.toml is gitignored)

# Run benchmarks. Which benchmarks run per suite is set in benchmarks.json;
# the model defaults to the `deepseek` section of models.toml.
uv run python -m scripts.function_calling          # both agent types
uv run python -m scripts.function_calling -a cave  # CaveAgent (Python code execution) only
uv run python -m scripts.function_calling -a json  # LiteLLM (JSON function calling) only
uv run python -m scripts.data_analysis             # CaveAgent only (no -a)
uv run python -m scripts.smart_home

# Common flags (all scripts): -m/--model, -b/--benchmark, --exp, --no-skip
uv run python -m scripts.function_calling -m gemini -b flight_booking --exp run1
uv run python -m scripts.data_analysis --no-skip   # ignore prior output, re-run all

# Tests
uv run pytest                                 # all tests
uv run pytest tests/test_validation.py        # one file
uv run pytest tests/test_validation.py::test_name -v   # one test
```

Models are defined in `models.toml` (gitignored secrets; copy from `models.toml.example`). Each `[section]` is a selectable model referenced by `--model <section>` (default `deepseek`); fields are `api_model`, `api_key`, optional `base_url`, `provider` (default `openai`, i.e. an OpenAI-compatible endpoint), `temperature`, and the per-model knobs `max_tokens`, `reasoning_effort` (`low|medium|high`), `extra_body` (arbitrary litellm passthrough), and a `[model.thinking]` table mapping `on`/`off` to an `extra_body` fragment. Configs are loaded/validated by `core/llm/`'s `ModelRegistry` (pydantic, `extra="forbid"`). `scripts/_common.py` turns a `ModelConfig` into the concrete CaveAgent / LiteLLM model + factory, forwarding the optional knobs to litellm (both agent paths route through it; `litellm.drop_params=True` so a model that rejects e.g. `temperature` doesn't error). The `--thinking on|off` flag selects the thinking variant (merged via `ModelConfig.resolve_extra_body`) and is folded into `exp_id` so on/off runs don't collide.

## Architecture

The central abstraction is in `core/agent.py`: every agent implements `Agent.run(query) -> AgentResponse`, and an `AgentFactory.create_agent(...)` builds one per conversation. `AgentResponse` carries `content`, `tool_calls`, `steps`, `code_snippets`, and `token_usage`. Everything downstream is written against these interfaces, never against a concrete agent — this is what lets one evaluator score both CaveAgent and JSON function calling.

**Adapters** (`adapters/`) wrap concrete implementations to that interface:
- `CaveAgentFactory` / `CaveAgentWrapper` — wraps CaveAgent. Since CaveAgent calls tools by executing generated Python (not by emitting structured tool calls), the wrapper can't read tool calls off an API response. Instead `core/tracker.py`'s `FunctionCallTracker` installs a `sys.setprofile` hook for the duration of `run()` and records every invocation of a target function name, reconstructing `ToolCall`s from live stack frames. This profiling-based capture is the load-bearing trick of the whole framework — keep it in mind when touching the CaveAgent path.
- `LitellmAgentFactory` / `LitellmAgentWrapper` — a real agentic loop over LiteLLM JSON function calling (call model → run returned tool calls → feed results back → repeat until no tool calls or `max_steps`). `variables`/`types` args are ignored here; only CaveAgent is stateful.

**Evaluation flow:** `runner.evaluate(factory, scenarios, output_file, resume=True)` is the entry point used by every script. It is **incremental/resumable** — by default it loads the existing output JSON and skips any scenario already present by name, writing results back after each scenario. Pass `resume=False` (scripts: `--no-skip`) to ignore the existing file and re-run everything. Output goes to `runs/<suite>/<exp_id>/<benchmark>.json`; reuse the same `--exp` to resume a prior run. On an infrastructure error (bad import, API/DB failure) the scenario is recorded as failed via `_make_error_result` and the run continues. It hands each scenario to `core/evaluator.py`'s `Evaluator`, which iterates conversations → turns, calls `agent.run()`, and aggregates metrics.

**Per-turn scoring** (in `Evaluator._evaluate_turn`) combines several independent signals:
- Function-call validation (`core/validation.py`): fuzzy-matches actual calls to `expected_function_calls`, counting missing calls / wrong argument values / wrong types / missing args. Matching is cost-based (`calculate_mismatch_cost`) and values are normalized before comparison (`normalize_value`, e.g. `2750.0 == "2750"`); types use `is_type_compatible` (int/float and numeric-strings are interchangeable). Set `strict_args: true` on an expected call to flag unexpected arguments.
- Variable access: `analyze_variable_access` AST-parses each emitted code snippet and checks `expected_variable_reads` / `expected_variable_writes` (only meaningful for CaveAgent, which produces code).
- Custom validator: an optional `validator` function gets `(response, runtime, turn, actual_calls)` and returns `ValidatorResult(success, message, variables_not_set=False)`. For CaveAgent it can reach final variable state via `runtime.retrieve("name")`; set `variables_not_set=True` to signal the agent never populated the required variables (refusal/infra) vs computed-but-wrong. For numeric checks use `core.validation.compare_numeric` (signed, NaN/None-safe) rather than hand-rolling `abs(a-b) > tol`.

A turn succeeds only if there are **zero** validation errors AND the custom validator passes.

**Scenarios = JSON + Python module.** A scenario is one JSON file (in `evals/<suite>/<name>.json`) plus the Python module it names in its `module` field (e.g. `evals.data_analysis.comparative_analysis`):
- The **JSON** defines structure: `conversations -> turns`, each turn with `query`, optional `validator` (name), `expected_function_calls`, `expected_variable_reads/writes`, optional `pre_turn_hook`. It may also override `description`/`requirements`.
- The **Python module** exports the callables and state: `tools` (list of functions), `variables` (`cave_agent.Variable`), `validators` (name→fn dict), `hooks` (name→fn dict, run before a turn to mutate runtime state / return a new query), `types`, and optional `description`/`requirements`. `BenchmarkScenario.from_module` pulls these by attribute name, with JSON values taking precedence over module attributes for description/requirements.

Each conversation gets a freshly created agent and a `deepcopy` of `variables`, so mutable state never leaks across conversations. Results are written to `runs/<suite>/<exp_id>/<name>.json`.

**Prompts:** CaveAgent runs are instructed to operate in a persistent Jupyter-like Python session and to do all computation via code (never fabricate results); see `core/prompts.py`. The JSON-function-calling system prompt lives in `adapters/litellm_adapter.py`.

## Conventions

- Adding a benchmark = add the `<name>.json` + sibling `.py` module under `evals/<suite>/`, then register `<name>` under its suite in `benchmarks.json` (the single source of truth for what each script runs; no per-script `BENCHMARKS` list anymore).
- Keep new agents conforming to the `Agent`/`AgentFactory` interface; do not special-case agent types inside `Evaluator` or `runner`.
- `recovered/` and `runs/` hold snapshots and generated output — not source; don't treat them as the live codebase.
