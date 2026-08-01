"""Shared helpers for the benchmark runner scripts.

Centralizes what used to be copy-pasted across function_calling.py /
data_analysis.py / smart_home.py:

- model selection from `models.toml` (via `core.llm.ModelRegistry`)
- benchmark selection from `benchmarks.json`
- exp_id-based output paths so runs are resumable (the runner skips
  scenarios already present in the output file; reuse the same exp_id to
  resume, or pass --no-skip to start fresh)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm

from core.llm import ModelConfig, ModelRegistry

# Both agent paths route through litellm; drop_params lets one config drive
# heterogeneous models without per-model branching — unsupported params are
# dropped, not errored. It only covers models litellm has metadata for, though:
# behind a custom gateway it cannot know what the model rejects, so a model that
# refuses `temperature` still 400s. That case is handled by
# `ModelConfig.supports_temperature` / `_temperature` below.
litellm.drop_params = True

# A benchmark sweep is long and every scenario costs an API call, so a single
# transient fault (rate limit, 5xx, a proxy closing an idle tunnel) should not
# be recorded as a model failure. litellm retries on its own classification of
# retryable errors; set here rather than per-call so both agent paths inherit it.
litellm.num_retries = 3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_TOML = PROJECT_ROOT / "models.toml"
BENCHMARKS_JSON = PROJECT_ROOT / "benchmarks.json"


def get_model(name: str) -> ModelConfig:
    """Resolve a model section from models.toml by name."""
    return ModelRegistry.load(MODELS_TOML).get(name)


def _temperature(cfg: ModelConfig) -> Optional[float]:
    """The temperature to send, or None for models that reject the parameter.

    Both agent paths hand this straight to litellm, which omits a None. Reasoning
    models reject the parameter, and so do a few others; see
    `ModelConfig.supports_temperature` for why `litellm.drop_params` is not enough.
    """
    if cfg.reasoning or not cfg.supports_temperature:
        return None
    return cfg.temperature


def _model_params(cfg: ModelConfig, thinking_mode: Optional[str]) -> Dict[str, Any]:
    """Optional litellm params shared by both agent paths (only the set ones)."""
    params: Dict[str, Any] = {}
    if cfg.max_tokens is not None:
        params["max_tokens"] = cfg.max_tokens
    if cfg.timeout is not None:
        params["timeout"] = cfg.timeout
    if cfg.reasoning_effort:
        params["reasoning_effort"] = cfg.reasoning_effort
    extra_body = cfg.resolve_extra_body(thinking_mode)
    if extra_body:
        params["extra_body"] = extra_body
    return params


def make_cave_factory(cfg: ModelConfig, thinking_mode: Optional[str] = None):
    """Build a CaveAgent factory (Python code execution) from a ModelConfig.

    cave_agent's LiteLLMModel forwards **kwargs to litellm.acompletion, so the
    optional reasoning/thinking params ride through unchanged.
    """
    from cave_agent.models import LiteLLMModel
    from adapters import CaveAgentFactory

    model = LiteLLMModel(
        model_id=cfg.api_model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=_temperature(cfg),
        custom_llm_provider=cfg.provider,
        **_model_params(cfg, thinking_mode),
    )
    return CaveAgentFactory(model)


def make_json_factory(cfg: ModelConfig, thinking_mode: Optional[str] = None):
    """Build a LiteLLM factory (JSON function calling) from a ModelConfig."""
    from adapters import LitellmAgentFactory, LitellmModel

    params = _model_params(cfg, thinking_mode)
    model = LitellmModel(
        model_id=cfg.api_model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=_temperature(cfg),
        provider=cfg.provider,
        max_tokens=params.get("max_tokens"),
        reasoning_effort=params.get("reasoning_effort"),
        extra_body=params.get("extra_body"),
    )
    return LitellmAgentFactory(model)


def resolve_benchmarks(suite: str, only: Optional[str] = None) -> List[str]:
    """Return the benchmark names for a suite from benchmarks.json.

    `only` narrows to a single benchmark (errors if it isn't registered).
    """
    registry = json.loads(BENCHMARKS_JSON.read_text())
    if suite not in registry:
        raise SystemExit(f"suite {suite!r} not in benchmarks.json; have: {sorted(registry)}")
    names = registry[suite]
    if only:
        if only not in names:
            raise SystemExit(f"benchmark {only!r} not in suite {suite!r}; available: {names}")
        return [only]
    return names


def load_scenarios(suite: str, name: str) -> list:
    """Load a benchmark's scenario list from evals/<suite>/<name>.json."""
    path = PROJECT_ROOT / "evals" / suite / f"{name}.json"
    return json.loads(path.read_text())


def output_path(suite: str, name: str, exp_id: str) -> str:
    """Stable, resumable output path: runs/<suite>/<exp_id>/<name>.json.

    Pure path computation — the parent dir is created by `runner.evaluate`,
    and resume/overwrite is controlled by its `resume` flag. No timestamp in
    the name, so re-running with the same exp_id resumes.
    """
    return str(PROJECT_ROOT / "runs" / suite / exp_id / f"{name}.json")
