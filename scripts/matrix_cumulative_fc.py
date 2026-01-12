"""Matrix Cumulative Benchmark Runner - JSON Function Calling (LiteLLM)"""

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

from adapters.litellm_adapter import LitellmAgentFactory, LitellmModel
from runner import evaluate


# Configuration
MODEL_ID = os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat")
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))


async def run_evaluation():
    model = LitellmModel(
        model_id=MODEL_ID,
        api_key=API_KEY,
        provider="deepseek",
        base_url=BASE_URL,
        temperature=TEMPERATURE,
    )
    factory = LitellmAgentFactory(model, max_steps=300)

    print(f"\n{'='*60}\nBenchmark: matrix_cumulative_fc (JSON Function Calling)\n{'='*60}")

    scenarios = json.loads(Path("./benchmarks/precision/matrix_cumulative_fc.json").read_text())
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = f"./results/precision_matrix_cumulative_fc/{MODEL_ID}_{timestamp}.json"

    await evaluate(factory, scenarios, output)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
