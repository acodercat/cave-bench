"""Main evaluation runner for agents.

This module provides the evaluate function which can run any agent
implementation that conforms to the AgentFactory interface.
"""

import json
import logging
import importlib
import warnings
from pathlib import Path
from typing import Dict, Any, List

from core.evaluator import Evaluator
from core.types import Conversation, ScenarioResult
from core.agent import AgentFactory

logger = logging.getLogger(__name__)


# Suppress Pydantic serialization warnings from LiteLLM response objects                                                                                                     
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic") 


async def evaluate(
    agent_factory: AgentFactory,
    scenarios: List[Dict[str, Any]],
    output_file: str
) -> List[ScenarioResult]:
    """
    Run evaluation on scenarios and save results.

    Supports incremental evaluation - skips already evaluated scenarios
    and resumes from where it left off.

    Args:
        agent_factory: Factory for creating agent instances
        scenarios: List of scenario definitions with expected outputs
        output_file: Path to save/load evaluation results (JSON format)

    Returns:
        List of ScenarioResult objects for all evaluated scenarios

    Example:
        >>> from cave_agent.models import LiteLLMModel
        >>> from adapters import CaveAgentFactory
        >>>
        >>> model = LiteLLMModel(model_id="gpt-4o", ...)
        >>> factory = CaveAgentFactory(model)
        >>> scenarios = json.load(open("evals/function_calling/weather.json"))
        >>> results = await evaluate(factory, scenarios, "runs/output.json")

    Example with LiteLLM (JSON function calling):
        >>> from adapters import LitellmAgentFactory, LitellmModel
        >>>
        >>> model = LitellmModel(model_id="gpt-4o", api_key="...", provider="openai")
        >>> factory = LitellmAgentFactory(model)
        >>> results = await evaluate(factory, scenarios, "runs/output.json")
    """
    # Load existing results if available
    existing_results = {}
    output_path = Path(output_file)

    if output_path.exists():
        with open(output_path, 'r') as f:
            existing_results = json.load(f)
        logger.info(f"Loaded {len(existing_results)} existing results from {output_file}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator with the factory
    evaluator = Evaluator(agent_factory)

    total_scenarios = len(scenarios)
    logger.info(f"Starting evaluation: {total_scenarios} scenarios")

    # Collect all results for return value
    all_results: List[ScenarioResult] = []

    # Run evaluation on each scenario
    for idx, scenario in enumerate(scenarios, 1):
        scenario_name = scenario.get('name', f"scenario_{idx}")

        # Skip if already evaluated
        if scenario_name in existing_results:
            logger.debug(f"[{idx}/{total_scenarios}] Skipping {scenario_name} (already evaluated)")
            continue

        logger.info(f"[{idx}/{total_scenarios}] Evaluating: {scenario_name}")

        # Load tools module
        tools_module = importlib.import_module(scenario['module'])

        # Convert conversations to typed objects
        typed_conversations = [
            Conversation.from_dict(conversation)
            for conversation in scenario['conversations']
        ]

        # Run evaluation (pass scenario dict for description/instructions)
        results = await evaluator.evaluate(
            scenario_name,
            tools_module,
            typed_conversations,
            json_config=scenario  # Pass full scenario dict for description/instructions extraction
        )

        # Collect result
        all_results.append(results)

        # Print summary metrics
        metrics = results.metrics
        avg_steps = metrics.total_steps / metrics.total_turns if metrics.total_turns > 0 else 0
        print(f"\nResults:")
        print(f"  Success Rate: {metrics.success_rate:.1%} ({metrics.successful_turns}/{metrics.total_turns})")
        print(f"  Failed Turns: {metrics.failed_turns}")
        print(f"  Total Steps: {metrics.total_steps}")
        print(f"  Avg Steps/Turn: {avg_steps:.1f}")
        print(f"  Token Usage:")
        print(f"    Prompt Tokens: {metrics.total_prompt_tokens:,}")
        print(f"    Completion Tokens: {metrics.total_completion_tokens:,}")
        print(f"    Total Tokens: {metrics.total_tokens:,}")

        # Save results incrementally
        existing_results[scenario_name] = results.to_dict()
        with open(output_path, 'w') as f:
            json.dump(existing_results, f, indent=2)

        print(f"\nSaved to: {output_file}")
        print(f"Progress: {idx}/{total_scenarios} scenarios completed")

    # Final summary
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Scenarios: {total_scenarios}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")

    return all_results
