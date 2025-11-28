"""
Blackjack Benchmark Runner

This script evaluates CaveAgent's ability to make step-by-step decisions
in a game of Blackjack. Unlike other benchmarks where the agent can plan
everything in one code block, this benchmark injects randomness between
turns, forcing genuine turn-by-turn reasoning.

Key Features:
- Pre-turn hooks modify game state (deal cards, process hits)
- Each turn requires ONE decision based on current visible state
- Agent cannot pre-plan entire game - must respond to random cards
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from cave_agent.models import LiteLLMModel
from runner import evaluate
from utils import load_model_config


benchmark_paths = [
    "/home/codercat/Desktop/Workplace/Lab/cave-bench/benchmarks/game/blackjack.json"
]


async def evaluate_model(model_name: str):
    """
    Evaluate a model on the blackjack benchmark.

    Args:
        model_name: Name of the model in models.toml
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Benchmark: Blackjack (Step-by-Step Decision Making)")
    print(f"{'='*60}\n")

    # Load model configuration
    model_config = load_model_config(model_name)
    print(f"Model config: {model_config}")

    # Create model instance
    model = LiteLLMModel(**model_config, custom_llm_provider='openai')

    # Run evaluation
    output_file = f"./results/blackjack_{model_config['model_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    benchmark_scenarios = []
    for benchmark_path in benchmark_paths:
        benchmark_scenario = json.loads(Path(benchmark_path).read_text())
        benchmark_scenarios.extend(benchmark_scenario)

    await evaluate(model, benchmark_scenarios, output_file)


async def evaluate_gemini():
    """Evaluate Gemini 3 model."""
    await evaluate_model("gemini3")


async def evaluate_kimi_k2():
    """Evaluate Kimi K2 model."""
    await evaluate_model("kimi-k2")


async def evaluate_qwen3_max():
    """Evaluate Qwen3 Max model."""
    await evaluate_model("qwen3-max")


if __name__ == "__main__":
    # Choose which model to evaluate
    asyncio.run(evaluate_qwen3_max())
