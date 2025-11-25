# cave-bench

Benchmarking framework for evaluating [CaveAgent](https://github.com/acodercat/cave-agent)'s tool-calling and code-generation capabilities.

## Installation

```bash
uv sync
```

## Quick Start

```python
import asyncio
from pathlib import Path
import json
from cave_agent.models import LiteLLMModel
from runner import evaluate

model = LiteLLMModel(
    model_id="gpt-4o",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"
)

benchmarks = json.loads(Path("./benchmarks/function_calling/benchmarks.json").read_text())

asyncio.run(evaluate(model, benchmarks, "./results/results.json"))
```

Or run directly:

```bash
python run_function_calling.py  # Function calling benchmarks
python run_data_analysis.py     # Data analysis benchmarks
```

## Benchmark Structure

### JSON Schema

```json
{
  "name": "scenario_name",
  "module": "benchmarks.data_analysis.MyDataset.my_analysis",
  "description": "Optional task description for the agent",
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
from core.types import BenchmarkTurn, ToolCall
import pandas as pd

df = pd.read_csv("path/to/dataset.csv")

def validate_q1(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    result = runtime.get_variable_value("result")
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

## Configuration

Model configs in `models.toml`:

```toml
[gpt-4o]
model_id = "gpt-4o"
api_key = "your-api-key"
base_url = "https://api.openai.com/v1"
```

## Contributing

Contributions are welcome! Please feel free to submit a PR.
For more details, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE) for details.
