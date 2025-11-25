"""Main benchmark runner for evaluating CaveAgent."""

import json
import importlib
from pathlib import Path
from typing import Dict, Any, List
from core.evaluator import BenchmarkEvaluator
from core.types import BenchmarkConversation
from cave_agent import Model


async def evaluate(
    model: Model,
    scenarios: List[Dict[str, Any]],
    results_file: str
) -> None:
    """
    Run evaluation on multiple scenarios and save results.

    This function supports incremental evaluation - it will skip scenarios
    that have already been evaluated and resume from where it left off.

    Args:
        model: The LLM model to use for evaluation
        scenarios: List of scenario definitions with expected outputs
        results_file: Path to save/load evaluation results (JSON format)

    Example:
        >>> from cave_agent.models import LiteLLMModel
        >>>
        >>> model = LiteLLMModel(model_id="gpt-4o", ...)
        >>> scenarios = json.load(open("benchmarks/function_calling/benchmarks.json"))
        >>> await evaluate(model, scenarios, "results.json")
    """
    # Load existing results if available
    existing_results = {}
    results_path = Path(results_file)

    if results_path.exists():
        with open(results_path, 'r') as f:
            existing_results = json.load(f)
        print(f"\nLoaded {len(existing_results)} existing results from {results_file}")

    # Ensure results directory exists
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    evaluator = BenchmarkEvaluator(model)

    # Print header
    total_scenarios = len(scenarios)
    print(f"\n{'='*60}")
    print(f"Starting Evaluation: {total_scenarios} scenarios")
    print(f"{'='*60}\n")

    # Run evaluation on each scenario
    for idx, scenario in enumerate(scenarios, 1):
        scenario_name = scenario.get('name', f"scenario_{idx}")

        # Skip if already evaluated
        if scenario_name in existing_results:
            print(f"[{idx}/{total_scenarios}] SKIP: {scenario_name} (already evaluated)")
            continue

        print(f"\n{'='*60}")
        print(f"[{idx}/{total_scenarios}] Evaluating: {scenario_name}")
        print(f"{'='*60}")

        # Load tools module
        tools_module = importlib.import_module(scenario['module'])

        # Convert conversations to typed objects
        typed_conversations = [
            BenchmarkConversation.from_dict(conversation)
            for conversation in scenario['conversations']
        ]

        # Run evaluation (pass scenario dict for description/instructions)
        results = await evaluator.evaluate(
            scenario_name,
            tools_module,
            typed_conversations,
            json_config=scenario  # Pass full scenario dict for description/instructions extraction
        )

        # Print summary metrics
        metrics = results.metrics
        print(f"\nResults:")
        print(f"  Success Rate: {metrics.success_rate:.1%} ({metrics.successful_turns}/{metrics.total_turns})")
        print(f"  Failed Turns: {metrics.failed_turns}")
        print(f"  Total Steps: {metrics.total_steps}")
        print(f"  Avg Steps/Turn: {metrics.total_steps/metrics.total_turns:.1f}" if metrics.total_turns > 0 else "")

        # Save results incrementally (convert to dict for JSON serialization)
        existing_results[scenario_name] = results.to_dict()
        with open(results_path, 'w') as f:
            json.dump(existing_results, f, indent=2)

        print(f"\nSaved to: {results_file}")
        print(f"Progress: {idx}/{total_scenarios} scenarios completed")

    # Final summary
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Scenarios: {total_scenarios}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}\n")