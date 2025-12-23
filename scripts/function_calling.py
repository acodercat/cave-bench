"""Function Calling Benchmark Runner"""

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
    "financial_portfolio_analysis",
    # "weather_query",
    # "recipe_planning",
    # "smart_home_automation",
    # "travel_planning",
    # "flight_booking",
    # "order_processing",
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

        scenarios = json.loads(Path(f"./benchmarks/function_calling/{name}.json").read_text())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"./results/function_calling_{name}/{MODEL_ID}_{timestamp}.json"

        await evaluate(factory, scenarios, output)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
