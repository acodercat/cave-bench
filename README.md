# cave-bench

Benchmarking framework for evaluating [CaveAgent](https://github.com/acodercat/cave-agent)'s tool-calling and code-generation capabilities.

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

Run benchmarks from the `scripts/` folder:

```bash
python scripts/function_calling.py  # Function calling benchmarks
python scripts/stateful.py          # Stateful benchmarks
python scripts/data_analysis.py     # Data analysis benchmarks
python scripts/domain.py            # Smart home, robot, game, etc.
```

Edit the `BENCHMARKS` list in each script to select which benchmarks to run.

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
    result = runtime.get_variable("result")
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

## Project Structure

```
cave-bench/
├── adapters/           # Agent adapters (CaveAgent, LiteLLM)
│   └── models/         # Model implementations (Gemini, Anthropic)
├── benchmarks/         # Benchmark definitions
│   ├── function_calling/
│   ├── stateful/
│   ├── data_analysis/
│   ├── smart_home/
│   ├── robot/
│   └── game/
├── core/               # Evaluation framework
├── scripts/            # Benchmark runners
├── results/            # Evaluation results
└── runner.py           # Core evaluation engine
```

## Contributing

Contributions are welcome! Please feel free to submit a PR.

## License

MIT License - see [LICENSE](LICENSE) for details.
