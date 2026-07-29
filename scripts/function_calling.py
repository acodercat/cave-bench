"""Function Calling Benchmark Runner.

Examples:
    uv run python -m scripts.function_calling                       # both agents, model=deepseek
    uv run python -m scripts.function_calling -a cave -m gemini     # CaveAgent only, gemini
    uv run python -m scripts.function_calling -b flight_booking     # one benchmark
    uv run python -m scripts.function_calling --exp run1            # custom exp_id (resumable)
    uv run python -m scripts.function_calling --no-skip             # force clean re-run
"""

import argparse
import asyncio

from scripts._common import (
    get_model,
    make_cave_factory,
    make_json_factory,
    resolve_benchmarks,
    load_scenarios,
    output_path,
)
from runner import evaluate

SUITE = "function_calling"


async def main(agent_type: str, model_name: str, exp: str, only: str, no_skip: bool, thinking: str):
    cfg = get_model(model_name)
    benchmarks = resolve_benchmarks(SUITE, only)
    exp_id = exp or model_name
    if thinking:
        exp_id = f"{exp_id}_think-{thinking}"

    runs = []
    if agent_type in ("cave", "all"):
        runs.append(("cave", make_cave_factory(cfg, thinking)))
    if agent_type in ("json", "all"):
        runs.append(("json", make_json_factory(cfg, thinking)))

    for tag, factory in runs:
        for name in benchmarks:
            print(f"\n{'='*60}\nBenchmark: {name} ({tag})\n{'='*60}")
            scenarios = load_scenarios(SUITE, name)
            output = output_path(SUITE, name, f"{exp_id}_{tag}")
            await evaluate(factory, scenarios, output, resume=not no_skip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Function Calling Benchmark Runner")
    parser.add_argument("--agent", "-a", choices=["cave", "json", "all"], default="all",
                        help="Agent type: cave (Python code), json (JSON function calling), or all")
    parser.add_argument("--model", "-m", default="deepseek",
                        help="Model section name from models.toml (default: deepseek)")
    parser.add_argument("--benchmark", "-b", default=None,
                        help="Run a single benchmark by name (default: all in suite)")
    parser.add_argument("--exp", default=None,
                        help="Experiment id for output dir; reuse to resume (default: model name)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Delete existing output and re-run from scratch")
    parser.add_argument("--thinking", choices=["on", "off"], default=None,
                        help="Select the model's thinking variant (needs a [model.thinking] table)")
    args = parser.parse_args()
    asyncio.run(main(args.agent, args.model, args.exp, args.benchmark, args.no_skip, args.thinking))
