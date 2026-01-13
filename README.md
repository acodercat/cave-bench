# cave-bench

Benchmarking framework for evaluating [CaveAgent](https://github.com/acodercat/cave-agent) tool calling, stateful management, and JSON-based tool calling.

## Installation

```bash
uv sync
```

## Configuration

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

```bash
# .env
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_MODEL_ID=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_TEMPERATURE=0.3
```

## Quick Start

Run benchmarks using module syntax:

```bash
# Function calling benchmarks
python -m scripts.function_calling           # Run both agent types
python -m scripts.function_calling -a cave   # CaveAgent (Python code execution)
python -m scripts.function_calling -a json   # LiteLLM (JSON function calling)

# Other benchmarks
python -m scripts.data_analysis     # Data analysis benchmarks
python -m scripts.smart_home        # Smart home benchmarks
```

Edit the `BENCHMARKS` list in each script to select which benchmarks to run.

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
