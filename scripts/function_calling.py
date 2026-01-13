"""Function Calling Benchmark Runner"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from cave_agent.models import LiteLLMModel
from adapters import CaveAgentFactory, LitellmAgentFactory, LitellmModel
from runner import evaluate


# Configuration
MODEL_ID = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat")
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))

BENCHMARKS = [
    "weather_query",
    "flight_booking",
]


async def run_cave():
    """Run evaluation with CaveAgent (Python code execution)."""
    model = LiteLLMModel(
        model_id=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=TEMPERATURE,
        custom_llm_provider='openai'
    )
    factory = CaveAgentFactory(model)

    for name in BENCHMARKS:
        print(f"\n{'='*60}\nBenchmark: {name} (cave)\n{'='*60}")

        scenarios = json.loads(Path(f"./evals/function_calling/{name}.json").read_text())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"./runs/function_calling/{name}/{MODEL_ID}_cave_{timestamp}.json"

        await evaluate(factory, scenarios, output)


async def run_json():
    """Run evaluation with LiteLLM (JSON function calling)."""
    model = LitellmModel(
        model_id=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=TEMPERATURE,
        provider='openai'
    )
    factory = LitellmAgentFactory(model)

    for name in BENCHMARKS:
        print(f"\n{'='*60}\nBenchmark: {name} (json)\n{'='*60}")

        scenarios = json.loads(Path(f"./evals/function_calling/{name}.json").read_text())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"./runs/function_calling/{name}/{MODEL_ID}_json_{timestamp}.json"

        await evaluate(factory, scenarios, output)


async def main(agent_type: str):
    if agent_type in ('cave', 'all'):
        await run_cave()
    if agent_type in ('json', 'all'):
        await run_json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Function Calling Benchmark Runner")
    parser.add_argument(
        '--agent', '-a',
        choices=['cave', 'json', 'all'],
        default='all',
        help='Agent type: cave (Python code), json (JSON function calling), or all (default)'
    )
    args = parser.parse_args()
    asyncio.run(main(args.agent))
