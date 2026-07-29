"""Main evaluation runner for agents.

This module provides the evaluate function which can run any agent
implementation that conforms to the AgentFactory interface.
"""

import json
import logging
import importlib
import os
import warnings
from pathlib import Path
from typing import Dict, Any, List

from core.evaluator import Evaluator
from core.results import result_has_infra_error
from core.types import (
    Conversation,
    ConversationResult,
    ScenarioMetrics,
    ScenarioResult,
    TurnMetrics,
    TurnResult,
)
from core.agent import AgentFactory

logger = logging.getLogger(__name__)


# Suppress Pydantic serialization warnings from LiteLLM response objects
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


def _json_fallback(obj: Any) -> str:
    """Bounded repr for values json.dump can't serialize.

    Result payloads are diagnostics; a truncated repr is far better than a
    TypeError that aborts the incremental save after the file was truncated.
    """
    r = repr(obj)
    return r if len(r) <= 2000 else r[:2000] + "…[truncated]"


def _make_error_result(scenario: Dict[str, Any], error_msg: str) -> ScenarioResult:
    """Build a failed ScenarioResult for a scenario that crashed on an
    infrastructure error (bad module import, API/DB failure).

    Lets the run continue past one broken scenario instead of aborting the
    whole pass; the failure is recorded (every turn failed, `error` set) so
    it shows up in the output rather than silently vanishing.
    """
    conv_results = []
    total_turns = 0
    for conv in scenario.get("conversations", []):
        turns = conv.get("turns", [])
        total_turns += len(turns)
        turn_results = [
            TurnResult(
                query=t.get("query", ""),
                reference_response="",
                actual_response="",
                expected_calls=[],
                actual_calls=[],
                validation_errors=[f"Infrastructure error: {error_msg}"],
                metrics=TurnMetrics(),
                success=False,
                error=error_msg,
            )
            for t in turns
        ]
        conv_results.append(ConversationResult(
            id=conv.get("id", "unknown"),
            turns=turn_results,
        ))

    return ScenarioResult(
        scenario=scenario.get("name", "unknown"),
        conversations=conv_results,
        metrics=ScenarioMetrics(total_turns=total_turns, failed_turns=total_turns),
    )


async def evaluate(
    agent_factory: AgentFactory,
    scenarios: List[Dict[str, Any]],
    output_file: str,
    resume: bool = True,
) -> List[ScenarioResult]:
    """
    Run evaluation on scenarios and save results.

    Supports incremental evaluation - skips already evaluated scenarios
    and resumes from where it left off.

    Args:
        agent_factory: Factory for creating agent instances
        scenarios: List of scenario definitions with expected outputs
        output_file: Path to save/load evaluation results (JSON format)
        resume: When True (default), load any existing output file and skip
            scenarios already in it. When False, ignore it and re-run all.

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
    # Load existing results to resume; when resume=False, start fresh and
    # overwrite the file (already-evaluated scenarios are re-run).
    existing_results = {}
    output_path = Path(output_file)

    if resume and output_path.exists():
        with open(output_path) as f:
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

        # Skip if already evaluated — but retry scenarios whose stored result
        # was frozen by an infrastructure error (API outage, bad import).
        # Without this, one transient failure would be permanently recorded
        # as a model failure, since resume is the default workflow here.
        existing = existing_results.get(scenario_name)
        if existing is not None and not result_has_infra_error({scenario_name: existing}):
            logger.debug(f"[{idx}/{total_scenarios}] Skipping {scenario_name} (already evaluated)")
            continue

        logger.info(f"[{idx}/{total_scenarios}] Evaluating: {scenario_name}")

        try:
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
        except Exception as e:
            # Infrastructure error (bad import, API/DB failure) — record as a
            # failed scenario and keep going instead of aborting the whole run.
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[{idx}/{total_scenarios}] {scenario_name} crashed: {error_msg}")
            print(f"\n  ⚠ ERROR: {error_msg[:200]}")
            results = _make_error_result(scenario, error_msg)

        # Collect result
        all_results.append(results)

        # Print summary metrics
        metrics = results.metrics
        avg_steps = metrics.total_steps / metrics.total_turns if metrics.total_turns > 0 else 0
        print("\nResults:")
        print(f"  Success Rate: {metrics.success_rate:.1%} ({metrics.successful_turns}/{metrics.total_turns})")
        print(f"  Failed Turns: {metrics.failed_turns}")
        print(f"  Total Steps: {metrics.total_steps}")
        print(f"  Avg Steps/Turn: {avg_steps:.1f}")
        print("  Token Usage:")
        print(f"    Prompt Tokens: {metrics.total_prompt_tokens:,}")
        print(f"    Completion Tokens: {metrics.total_completion_tokens:,}")
        print(f"    Total Tokens: {metrics.total_tokens:,}")

        # Save results incrementally. Write-to-temp + atomic rename so a
        # serialization error or crash mid-dump can never truncate results
        # already on disk; `default=_json_fallback` keeps non-JSON values
        # (numpy scalars, DataFrames in tool-call args) from aborting the dump.
        existing_results[scenario_name] = results.to_dict()
        tmp_path = output_path.with_suffix(".json.tmp")
        with open(tmp_path, 'w') as f:
            json.dump(existing_results, f, indent=2, default=_json_fallback)
        os.replace(tmp_path, output_path)

        print(f"\nSaved to: {output_file}")
        print(f"Progress: {idx}/{total_scenarios} scenarios completed")

    # Final summary
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Scenarios: {total_scenarios}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")

    return all_results
