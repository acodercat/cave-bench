# cave-bench

Benchmarking framework for evaluating [CaveAgent](https://github.com/acodercat/cave-agent) tool calling, stateful management, and JSON-based tool calling.

## Installation

```bash
uv sync
```

## Configuration

Models are defined in `models.toml` (gitignored — it holds secrets). Copy the
template and fill in your keys:

```bash
cp models.toml.example models.toml
```

```toml
# models.toml — each [section] is a selectable model (--model <section>)
[deepseek]
api_model = "deepseek-chat"
api_key = "your-api-key"
base_url = "https://api.deepseek.com/v1"
provider = "openai"
temperature = 0.3
```

## Quick Start

```bash
# Function calling benchmarks (model defaults to `deepseek`)
uv run python -m scripts.function_calling            # both agent types
uv run python -m scripts.function_calling -a cave    # CaveAgent (Python code execution)
uv run python -m scripts.function_calling -a json    # LiteLLM (JSON function calling)

# Other suites
uv run python -m scripts.data_analysis
uv run python -m scripts.smart_home

# Pick a model, a single benchmark, a named experiment
uv run python -m scripts.function_calling -m gemini -b flight_booking --exp run1

# Thinking on/off axis (needs a [model.thinking] table in models.toml).
# The mode is folded into exp_id so on/off runs don't collide.
uv run python -m scripts.data_analysis -m qwen-thinking --thinking on
uv run python -m scripts.data_analysis -m qwen-thinking --thinking off
```

Per-model knobs in `models.toml` (all optional, forwarded to litellm): `max_tokens`, `reasoning_effort` (`low|medium|high`), `extra_body` (arbitrary passthrough), and a `[model.thinking]` table mapping `on`/`off` to an `extra_body` fragment. See `models.toml.example` for reasoning-model and thinking-axis templates.

Which benchmarks run per suite is defined in `benchmarks.json`. Results are
written to `runs/<suite>/<exp_id>/<benchmark>.json`. Reuse the same `--exp`
to **resume** (already-evaluated scenarios are skipped); pass `--no-skip` to
force a clean re-run.

## Benchmark Structure

### JSON Schema

```json
{
  "name": "scenario_name",
  "module": "evals.data_analysis.MyDataset.my_analysis",
  "requirements": "Optional task requirements",
  "conversations": [
    {
      "id": "test_1",
      "turns": [
        {
          "query": "Analyze the dataset...",
          "validator": "validate_q1",
          "expected_variable_reads": ["df"],
          "expected_variable_writes": ["result"]
        }
      ]
    }
  ]
}
```

### Python Module

```python
from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import Turn, ToolCall
import pandas as pd

df = pd.read_csv("path/to/dataset.csv")

def validate_q1(
    response: str,
    runtime: PythonRuntime,
    turn: Turn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    result = runtime.retrieve("result")
    if result == expected_value:
        return ValidatorResult(True, "Correct!")
    return ValidatorResult(False, f"Expected {expected_value}, got {result}")

tools = []
variables = [Variable("df", df, "Dataset description")]
validators = {"validate_q1": validate_q1}
```

## Metrics

- **Success Rate**: Percentage of successful turns
- **Function Calls**: Missing calls, wrong argument types/values
- **Variables**: Missing reads/writes
- **Steps**: Total steps taken
- **Tokens**: Consumed Tokens

## Contributing

Contributions are welcome! Please feel free to submit a PR.

## License

MIT License - see [LICENSE](LICENSE) for details.
