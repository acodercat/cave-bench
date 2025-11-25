"""
Example: Evaluating CaveAgent on function calling benchmarks.

This script demonstrates how to benchmark CaveAgent with different LLM models
on the function calling scenarios.
"""

import asyncio
import json
from pathlib import Path

from cave_agent.models import LiteLLMModel
from runner import evaluate
from utils import load_model_config


# Load benchmark scenarios
SCENARIOS_PATH = Path("./benchmarks/function_calling/benchmarks.json")
ground_truths = json.loads(SCENARIOS_PATH.read_text())


async def evaluate_model(model_name: str, output_prefix: str):
    """
    Evaluate a model on function calling benchmarks.

    Args:
        model_name: Name of the model in models.toml
        output_prefix: Prefix for the output results file
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}\n")

    # Load model configuration
    model_config = load_model_config(model_name)
    print(f"Model config: {model_config['model_id']}")

    # Create model instance
    model = LiteLLMModel(**model_config, custom_llm_provider='openai')

    # Run evaluation
    output_file = f"./results/{output_prefix}_{model_config['model_id']}.json"
    await evaluate(model, ground_truths, output_file)

    print(f"\nResults saved to: {output_file}")


async def evaluate_gemini3():
    """Evaluate Gemini model."""
    await evaluate_model("gemini3", "function_calling")

async def evaluate_kimi_k2():
    """Evaluate Kimi K2 model."""
    await evaluate_model("kimi-k2", "function_calling")


if __name__ == "__main__":
    # Choose which model to evaluate
    asyncio.run(evaluate_kimi_k2())