"""Stateful Benchmark Runner"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Setup
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)
load_dotenv()

from cave_agent.models import LiteLLMModel
from adapters.cave_agent_adapter import CaveAgentFactory
from runner import evaluate


# Configuration
MODEL_ID = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat")
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))

BENCHMARKS = [
    "simple_types",
    # "object_types",
    # "scientific_types",
    # "multi_turn_5",
    # "multi_turn_10",
    # "multi_turn_15",
    # "multi_turn_20",
]


async def run_evaluation():
    model = LiteLLMModel(
        model_id=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=TEMPERATURE,
        custom_llm_provider='openai'
    )
    factory = CaveAgentFactory(model)

    for name in BENCHMARKS:
        print(f"\n{'='*60}\nBenchmark: {name}\n{'='*60}")

        scenarios = json.loads(Path(f"./benchmarks/stateful/{name}.json").read_text())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"./results/stateful_{name}/{MODEL_ID}_{timestamp}.json"

        await evaluate(factory, scenarios, output)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
