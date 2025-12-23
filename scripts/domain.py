"""Domain-Specific Benchmark Runner (smart_home, robot, game, etc.)"""

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
    "smart_home/work_from_home_day",
    # "smart_home/basic_usage",
    # "robot/grid_navigation_robot",
    # "game/blackjack/blackjack",
    # "chemistry_lab/chemistry_lab_benchmarks",
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

        scenarios = json.loads(Path(f"./benchmarks/{name}.json").read_text())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = name.replace("/", "_")
        output = f"./results/{safe_name}/{MODEL_ID}_{timestamp}.json"

        await evaluate(factory, scenarios, output)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
