"""Stock Benchmark Runner - CodeAct Version"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)
load_dotenv()

from cave_agent.models import LiteLLMModel
from adapters.cave_agent_adapter import CaveAgentFactory
from runner import evaluate

MODEL_ID = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat")
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))

BENCHMARK_DIR = Path("./benchmarks/stock_analysis")


async def run_evaluation(benchmark_type: str = "all"):
    model = LiteLLMModel(
        model_id=MODEL_ID, api_key=API_KEY, base_url=BASE_URL,
        temperature=TEMPERATURE, custom_llm_provider='openai'
    )
    factory = CaveAgentFactory(model)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if benchmark_type == "all":
        types = ["query", "analysis", "viz"]
    else:
        types = [benchmark_type]

    for btype in types:
        print(f"\n{'='*60}\nStock Benchmark (CodeAct) - {btype}\n{'='*60}")
        scenarios = json.loads((BENCHMARK_DIR / f"stock_benchmark_codeact_{btype}.json").read_text())
        output = f"./results/stock_benchmark_codeact/{MODEL_ID}_{btype}_{timestamp}.json"
        await evaluate(factory, scenarios, output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["query", "analysis", "viz", "all"], default="all")
    args = parser.parse_args()
    asyncio.run(run_evaluation(args.type))
