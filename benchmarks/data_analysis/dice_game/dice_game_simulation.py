"""
This scenario is based on a statistical problem involving a modified dice game.
The goal is to use a Monte Carlo simulation to estimate the winning odds and other statistics.

Game Rules:
1.  The player rolls 3 standard six-sided dice.
2.  Initial Roll:
    -   Win on a total of 7 or 11.
    -   Lose on a total of 3, 4, 5, 16, 17, or 18.
    -   Any other total becomes the "point," and the player re-rolls.
3.  Re-rolls:
    -   Win by rolling the "point" number again.
    -   Win by rolling a 7 or 11.
    -   Lose by rolling a 3, 4, 5, 16, 17, or 18.
    -   Otherwise, continue re-rolling.
4.  A game concludes in a loss if it doesn't resolve within 50 rolls.
5.  The simulation should run for 5000 games (outcomes).

Questions to Answer (with correct answer buckets):
1.  What is the probability of winning? [D: >70%]
2.  What is the average number of rolls per game? [B: 2.7-2.9]
3.  What is the new win probability if rolling an 8 or 9 on the initial roll is also a loss? [C: 55%-65%]
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall

tools = []

num_simulations = 5000
max_rolls = 50

def validate_q1(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates the calculated win probability and average rolls, providing specific
    feedback for four distinct cases.
    """
    win_probability = runtime.get_variable_value("win_probability")
    avg_rolls = runtime.get_variable_value("avg_rolls")

    prob_correct = win_probability > 0.70
    rolls_correct = 2.7 <= avg_rolls < 2.9

    # Case 1: Both are correct (Success)
    if prob_correct and rolls_correct:
        return ValidatorResult(
            success=True,
            message="Correct. The win probability is greater than 70% and the average number of rolls is between 2.7 and 2.9."
        )

    # Case 2: Win probability is correct, but average rolls is incorrect
    elif prob_correct and not rolls_correct:
        return ValidatorResult(
            success=False,
            message=f"Incorrect. The average number of rolls ({avg_rolls:.2f}) is outside the expected 2.7-2.9 range, but the win probability was correct."
        )

    # Case 3: Win probability is incorrect, but average rolls is correct
    elif not prob_correct and rolls_correct:
        return ValidatorResult(
            success=False,
            message=f"Incorrect. The win probability ({win_probability:.1%}) is not in the expected >70% range, but the average number of rolls was correct."
        )
        
    # Case 4: Both are incorrect
    else:
        return ValidatorResult(
            success=False,
            message=f"Incorrect. The win probability ({win_probability:.1%}) is not in the expected >70% range, and the average rolls ({avg_rolls:.2f}) is outside the 2.7-2.9 range."
        )

def validate_q2(response: str, runtime: PythonRuntime, turn: BenchmarkTurn, actual_calls: List[ToolCall]) -> ValidatorResult:
    """
    Validates if the new win probability (with modified rules) is between 55% and 65%.
    """
    new_win_probability = runtime.get_variable_value("new_win_probability")
    if 0.55 <= new_win_probability < 0.65:
        return ValidatorResult(success=True, message="Correct. The new win probability is in the 55%-65% range.")
    else:
        return ValidatorResult(success=False, message=f"Incorrect. The calculated probability was {new_win_probability:.1%}, which is outside the expected 55%-65% range.")

variables = [
    Variable(
        name="num_simulations",
        value=num_simulations,
        description="The number of outcomes to simulate for the Monte Carlo analysis. example: 5000, you should get the number of simulations in this variable"
    ),
    Variable(
        name="max_rolls",
        value=max_rolls,
        description="The maximum number of rolls allowed in a single game before it's considered a loss. example: 50, you should get the max rolls in this variable"
    ),
    Variable(
        name="win_probability",
        value=0.0,
        description="Stores the calculated probability of winning the original game. example: 0.736"
    ),
    Variable(
        name="avg_rolls",
        value=0.0,
        description="Stores the calculated average number of rolls per game for the original game. example: 2.8"
    ),
    Variable(
        name="new_win_probability",
        value=0.0,
        description="Stores the calculated win probability for the modified game (where 8 and 9 are initial losing numbers). example: 0.597"
    )
]

validators = {
    "validate_q1": validate_q1,
    "validate_q2": validate_q2
}

# Usage example
if __name__ == "__main__":
    pass