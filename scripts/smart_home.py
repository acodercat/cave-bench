"""Smart Home Benchmark Runner (CaveAgent).

Examples:
    uv run python -m scripts.smart_home                       # model=deepseek
    uv run python -m scripts.smart_home -m gemini --exp run1
    uv run python -m scripts.smart_home -b evening_home_routine --no-skip
"""

import argparse
import asyncio

from scripts._common import (
    get_model,
    make_cave_factory,
    resolve_benchmarks,
    load_scenarios,
    output_path,
)
from runner import evaluate

SUITE = "smart_home"


async def main(model_name: str, exp: str, only: str, no_skip: bool, thinking: str):
    cfg = get_model(model_name)
    factory = make_cave_factory(cfg, thinking)
    benchmarks = resolve_benchmarks(SUITE, only)
    exp_id = exp or model_name
    if thinking:
        exp_id = f"{exp_id}_think-{thinking}"

    for name in benchmarks:
        print(f"\n{'='*60}\nBenchmark: {name}\n{'='*60}")
        scenarios = load_scenarios(SUITE, name)
        output = output_path(SUITE, name, exp_id)
        await evaluate(factory, scenarios, output, resume=not no_skip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Home Benchmark Runner")
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
    asyncio.run(main(args.model, args.exp, args.benchmark, args.no_skip, args.thinking))
