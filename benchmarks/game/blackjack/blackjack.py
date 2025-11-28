"""
Blackjack Benchmark - Step-by-Step Decision Making

Tests CaveAgent's ability to:
- Make decisions based on current state (not pre-plan everything)
- Respond to randomness injected between turns
- Apply basic strategy reasoning
- Handle uncertainty in outcomes

Key Innovation: Each turn is ONE decision. Randomness (card draws) happens
BETWEEN turns via pre_turn_hooks, forcing the agent to genuinely reason
at each step rather than writing a loop that handles everything.
"""

import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from cave_agent.python_runtime import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# CARD AND DECK
# ============================================================================

class Suit(Enum):
    HEARTS = "Hearts"
    DIAMONDS = "Diamonds"
    CLUBS = "Clubs"
    SPADES = "Spades"


class Rank(Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


@dataclass
class Card:
    """A playing card."""
    rank: Rank
    suit: Suit

    def value(self) -> int:
        """Get the blackjack value of this card (Ace = 11, face cards = 10)."""
        if self.rank == Rank.ACE:
            return 11
        elif self.rank in [Rank.JACK, Rank.QUEEN, Rank.KING]:
            return 10
        else:
            return int(self.rank.value)

    def __repr__(self):
        return f"{self.rank.value}{self.suit.value[0]}"


class Deck:
    """A deck of playing cards."""

    def __init__(self, num_decks: int = 1, seed: int = None):
        """Initialize deck with optional random seed for reproducibility."""
        self.num_decks = num_decks
        self.seed = seed
        self.cards: List[Card] = []
        self.reset()

    def reset(self):
        """Reset and shuffle the deck."""
        self.cards = []
        for _ in range(self.num_decks):
            for suit in Suit:
                for rank in Rank:
                    self.cards.append(Card(rank, suit))

        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(self.cards)

    def draw(self) -> Card:
        """Draw a card from the deck."""
        if not self.cards:
            self.reset()
        return self.cards.pop()

    def __len__(self):
        return len(self.cards)


# ============================================================================
# HAND
# ============================================================================

@dataclass
class Hand:
    """A blackjack hand."""
    cards: List[Card] = field(default_factory=list)

    def add_card(self, card: Card):
        """Add a card to the hand."""
        self.cards.append(card)

    def get_value(self) -> int:
        """
        Calculate hand value, handling Aces optimally.
        Aces count as 11 unless that would bust, then they count as 1.
        """
        value = sum(card.value() for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == Rank.ACE)

        # Convert Aces from 11 to 1 as needed to avoid bust
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1

        return value

    def is_bust(self) -> bool:
        """Check if hand is bust (over 21)."""
        return self.get_value() > 21

    def is_blackjack(self) -> bool:
        """Check if hand is a natural blackjack (21 with 2 cards)."""
        return len(self.cards) == 2 and self.get_value() == 21

    def is_soft(self) -> bool:
        """Check if hand is soft (has an Ace counted as 11)."""
        value = sum(card.value() for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == Rank.ACE)

        # Count how many aces we need to convert
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1

        # If we still have aces counted as 11, it's soft
        return aces > 0 and self.get_value() <= 21

    def describe(self) -> str:
        """Get a description of the hand."""
        cards_str = ", ".join(str(card) for card in self.cards)
        value = self.get_value()
        soft = " (soft)" if self.is_soft() else ""
        return f"[{cards_str}] = {value}{soft}"

    def clear(self):
        """Clear the hand."""
        self.cards = []

    def __repr__(self):
        return self.describe()


# ============================================================================
# GAME STATE
# ============================================================================

class GamePhase(Enum):
    WAITING = "waiting"      # Before game starts
    PLAYER_TURN = "player_turn"  # Player making decisions
    DEALER_TURN = "dealer_turn"  # Dealer revealing and drawing
    FINISHED = "finished"    # Game over


@dataclass
class GameState:
    """
    Complete blackjack game state.

    The agent interacts with this to make decisions.
    """
    deck: Deck
    player_hand: Hand
    dealer_hand: Hand
    phase: GamePhase = GamePhase.WAITING
    player_action: Optional[str] = None  # Last action taken: "hit" or "stand"
    result: Optional[str] = None  # "win", "lose", "push", "blackjack"
    message: str = ""
    turn_number: int = 0

    def get_visible_dealer_card(self) -> Card:
        """Get the dealer's face-up card (first card)."""
        if self.dealer_hand.cards:
            return self.dealer_hand.cards[0]
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get current game status for the agent."""
        dealer_visible = self.get_visible_dealer_card()
        return {
            "phase": self.phase.value,
            "player_hand": self.player_hand.describe(),
            "player_value": self.player_hand.get_value(),
            "player_is_soft": self.player_hand.is_soft(),
            "dealer_showing": str(dealer_visible) if dealer_visible else "None",
            "dealer_showing_value": dealer_visible.value() if dealer_visible else 0,
            "player_action": self.player_action,
            "result": self.result,
            "message": self.message,
            "turn_number": self.turn_number
        }

    def __repr__(self):
        return f"GameState(phase={self.phase.value}, player={self.player_hand}, dealer_showing={self.get_visible_dealer_card()})"


# ============================================================================
# GAME ACTIONS (What agent can call)
# ============================================================================

# Global game state - will be initialized in create_game()
game: GameState = None


def hit() -> Dict[str, Any]:
    """
    Take another card.

    Returns:
        dict with:
        - success: bool
        - message: str
        - new_card: str (the card drawn)
        - hand_value: int (new hand value)
        - is_bust: bool
    """
    global game

    if game.phase != GamePhase.PLAYER_TURN:
        return {
            "success": False,
            "message": f"Cannot hit: game phase is {game.phase.value}",
            "new_card": None,
            "hand_value": game.player_hand.get_value(),
            "is_bust": game.player_hand.is_bust()
        }

    # Record the action
    game.player_action = "hit"

    # Note: The actual card draw happens in the pre_turn_hook of the NEXT turn
    # This allows the agent to see the result and decide again

    return {
        "success": True,
        "message": "You chose to HIT. A card will be drawn.",
        "new_card": "pending",  # Will be revealed next turn
        "hand_value": game.player_hand.get_value(),
        "is_bust": False
    }


def stand() -> Dict[str, Any]:
    """
    Keep current hand and end player turn.

    Returns:
        dict with:
        - success: bool
        - message: str
        - final_hand_value: int
    """
    global game

    if game.phase != GamePhase.PLAYER_TURN:
        return {
            "success": False,
            "message": f"Cannot stand: game phase is {game.phase.value}",
            "final_hand_value": game.player_hand.get_value()
        }

    # Record the action
    game.player_action = "stand"
    game.phase = GamePhase.DEALER_TURN

    return {
        "success": True,
        "message": "You chose to STAND. Dealer will now play.",
        "final_hand_value": game.player_hand.get_value()
    }


def get_game_state() -> Dict[str, Any]:
    """
    Get current game state.

    Returns:
        dict with current game status
    """
    global game
    return game.get_status()


def get_basic_strategy_hint() -> str:
    """
    Get a hint based on basic blackjack strategy.

    Returns:
        str: Recommended action based on basic strategy
    """
    global game

    player_value = game.player_hand.get_value()
    dealer_up = game.get_visible_dealer_card().value() if game.get_visible_dealer_card() else 0
    is_soft = game.player_hand.is_soft()

    # Simplified basic strategy
    if player_value >= 17:
        return "STAND - You have 17 or higher"
    elif player_value <= 11:
        return "HIT - You cannot bust with 11 or less"
    elif is_soft:
        if player_value <= 17:
            return "HIT - Soft hand, safe to take a card"
        else:
            return "STAND - Soft 18+ is strong"
    else:
        # Hard hand 12-16
        if dealer_up >= 7:
            return "HIT - Dealer shows strong card (7+), you need to improve"
        elif dealer_up <= 6:
            return "STAND - Dealer shows weak card (2-6), let them bust"
        else:
            return "HIT - Borderline, but hitting is slightly better"


# ============================================================================
# GAME SETUP
# ============================================================================

def create_game(seed: int = None) -> GameState:
    """Create a new game with optional seed for reproducibility."""
    global game

    deck = Deck(num_decks=1, seed=seed)
    player_hand = Hand()
    dealer_hand = Hand()

    game = GameState(
        deck=deck,
        player_hand=player_hand,
        dealer_hand=dealer_hand,
        phase=GamePhase.WAITING
    )

    return game


# Initialize game with a seed for reproducible tests
game = create_game(seed=42)


# ============================================================================
# PRE-TURN HOOKS (Inject randomness between turns)
# ============================================================================

def hook_initial_deal(runtime: PythonRuntime, turn: BenchmarkTurn) -> str:
    """
    Hook for Turn 1: Deal initial cards.

    - Deals 2 cards to player
    - Deals 2 cards to dealer (one face down)
    - Returns dynamic query with current state
    """
    global game

    # Reset game state
    game.player_hand.clear()
    game.dealer_hand.clear()
    game.phase = GamePhase.PLAYER_TURN
    game.player_action = None
    game.result = None
    game.turn_number = 1

    # Deal initial cards
    game.player_hand.add_card(game.deck.draw())
    game.dealer_hand.add_card(game.deck.draw())  # Face up
    game.player_hand.add_card(game.deck.draw())
    game.dealer_hand.add_card(game.deck.draw())  # Face down

    # Update runtime variable
    runtime.update_variable("game", game)

    # Check for blackjack
    if game.player_hand.is_blackjack():
        game.message = "BLACKJACK! You have 21 with your first two cards!"
    else:
        game.message = "Cards dealt. Your turn to decide."

    # Generate dynamic query
    player_desc = game.player_hand.describe()
    dealer_showing = game.get_visible_dealer_card()

    query = f"""The cards have been dealt!

Your hand: {player_desc}
Dealer showing: {dealer_showing} (value: {dealer_showing.value()})

You must decide: HIT (take another card) or STAND (keep current hand).

Consider:
- Your hand value is {game.player_hand.get_value()}
- Dealer's visible card is {dealer_showing.value()}
- If you go over 21, you bust and lose

Call hit() to take another card, or stand() to keep your hand.
What is your decision?"""

    return query


def hook_after_hit(runtime: PythonRuntime, turn: BenchmarkTurn) -> str:
    """
    Hook for Turn 2+: Process previous hit action and deal new card.

    - If player chose HIT, draw a card
    - If player chose STAND, skip this turn
    - Returns dynamic query with new state
    """
    global game

    game.turn_number += 1

    # Check previous action
    if game.player_action == "stand":
        # Player stood, move to dealer phase
        game.phase = GamePhase.DEALER_TURN
        runtime.update_variable("game", game)

        return f"""You chose to STAND with {game.player_hand.describe()}.

The dealer will now reveal their hidden card and play.
This turn requires no action from you - just observe the result.
Call get_game_state() to see the current state."""

    elif game.player_action == "hit":
        # Player hit, draw a card
        new_card = game.deck.draw()
        game.player_hand.add_card(new_card)

        runtime.update_variable("game", game)

        # Check if bust
        if game.player_hand.is_bust():
            game.phase = GamePhase.FINISHED
            game.result = "lose"
            game.message = f"BUST! You drew {new_card} and went over 21."

            return f"""You drew: {new_card}

Your hand is now: {game.player_hand.describe()}

BUST! You went over 21 and lost the hand.
Game over. Call get_game_state() to see final result."""

        else:
            # Still in play
            dealer_showing = game.get_visible_dealer_card()

            return f"""You drew: {new_card}

Your hand is now: {game.player_hand.describe()}
Dealer still showing: {dealer_showing} (value: {dealer_showing.value()})

Your hand value is now {game.player_hand.get_value()}.
Do you want to HIT again or STAND?

Call hit() to take another card, or stand() to keep your hand."""

    else:
        # No previous action (shouldn't happen)
        return turn.query


def hook_dealer_plays(runtime: PythonRuntime, turn: BenchmarkTurn) -> str:
    """
    Hook for final turn: Dealer reveals and plays, determine winner.
    """
    global game

    game.turn_number += 1

    if game.phase == GamePhase.FINISHED:
        # Game already ended (player bust)
        return f"""The game has ended.
Result: {game.result}
Your final hand: {game.player_hand.describe()}
Call get_game_state() to see the final state."""

    # Dealer reveals and plays
    game.phase = GamePhase.DEALER_TURN

    # Dealer draws until 17 or higher
    while game.dealer_hand.get_value() < 17:
        game.dealer_hand.add_card(game.deck.draw())

    # Determine winner
    player_value = game.player_hand.get_value()
    dealer_value = game.dealer_hand.get_value()

    if game.dealer_hand.is_bust():
        game.result = "win"
        game.message = "Dealer busts! You win!"
    elif player_value > dealer_value:
        game.result = "win"
        game.message = "You win! Your hand beats the dealer."
    elif player_value < dealer_value:
        game.result = "lose"
        game.message = "Dealer wins. Better luck next time."
    else:
        game.result = "push"
        game.message = "Push! It's a tie."

    game.phase = GamePhase.FINISHED
    runtime.update_variable("game", game)

    return f"""Dealer reveals their hand and plays...

Dealer's full hand: {game.dealer_hand.describe()}
Your hand: {game.player_hand.describe()}

RESULT: {game.result.upper()}
{game.message}

Call get_game_state() to see the final game state and confirm the result."""


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_initial_decision(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Validate Turn 1: Player made a decision (hit or stand).
    """
    game = runtime.get_variable("game")

    # Check that player took an action
    if game.player_action is None:
        return ValidatorResult(False, "Player did not take any action (hit or stand)")

    if game.player_action not in ["hit", "stand"]:
        return ValidatorResult(False, f"Invalid action: {game.player_action}")

    # Check that action function was called
    action_calls = [c for c in actual_calls if c.function in ["hit", "stand"]]
    if not action_calls:
        return ValidatorResult(False, "Neither hit() nor stand() was called")

    return ValidatorResult(True, f"Player chose to {game.player_action.upper()}")


def validate_continued_play(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Validate Turn 2: Player responded appropriately to new card.
    """
    game = runtime.get_variable("game")

    # If game ended (bust), check that player acknowledged
    if game.phase == GamePhase.FINISHED and game.result == "lose":
        # Player busted, just verify they saw the result
        state_calls = [c for c in actual_calls if c.function == "get_game_state"]
        if state_calls:
            return ValidatorResult(True, "Player acknowledged bust")
        # Even without explicit call, response should mention bust
        if "bust" in response.lower() or "over 21" in response.lower():
            return ValidatorResult(True, "Player acknowledged bust in response")
        return ValidatorResult(True, "Game ended - player busted")

    # If still playing, check for action
    if game.phase == GamePhase.PLAYER_TURN:
        if game.player_action is None:
            return ValidatorResult(False, "Player did not take action")
        return ValidatorResult(True, f"Player chose to {game.player_action.upper()}")

    # If dealer's turn, player stood
    if game.phase == GamePhase.DEALER_TURN:
        return ValidatorResult(True, "Player stood, dealer's turn")

    return ValidatorResult(True, "Valid game progression")


def validate_game_complete(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Validate final turn: Game completed and result acknowledged.
    """
    game = runtime.get_variable("game")

    # Game should be finished
    if game.phase != GamePhase.FINISHED:
        return ValidatorResult(False, f"Game not finished, phase is {game.phase.value}")

    # Result should be set
    if game.result is None:
        return ValidatorResult(False, "Game result not determined")

    # Response should mention the result
    result_mentioned = (
        game.result in response.lower() or
        "win" in response.lower() or
        "lose" in response.lower() or
        "bust" in response.lower() or
        "push" in response.lower() or
        "tie" in response.lower()
    )

    if not result_mentioned:
        # Check if get_game_state was called
        state_calls = [c for c in actual_calls if c.function == "get_game_state"]
        if not state_calls:
            return ValidatorResult(False, "Result not acknowledged in response")

    return ValidatorResult(True, f"Game complete: {game.result}")


# ============================================================================
# EXPORTS
# ============================================================================

tools = [
    hit,
    stand,
    get_game_state,
    get_basic_strategy_hint
]

variables = [
    Variable(
        "game", game,
        "Current blackjack game state. Use get_game_state() to see current status. "
        "Call hit() to take a card or stand() to keep your hand."
    )
]

types = [
    Type(GameState),
    Type(Hand),
    Type(Card),
]

validators = {
    "validate_initial_decision": validate_initial_decision,
    "validate_continued_play": validate_continued_play,
    "validate_game_complete": validate_game_complete,
}

hooks = {
    "hook_initial_deal": hook_initial_deal,
    "hook_after_hit": hook_after_hit,
    "hook_dealer_plays": hook_dealer_plays,
}

description = """You are playing a game of Blackjack (21) against a dealer.

Game Rules:
- Goal: Get as close to 21 as possible without going over
- Number cards (2-10) are worth their face value
- Face cards (J, Q, K) are worth 10
- Aces are worth 11 (or 1 if 11 would bust)
- If you go over 21, you BUST and lose immediately
- Dealer must hit until they have 17 or higher

Your Options:
- HIT: Take another card (risk busting if you go over 21)
- STAND: Keep your current hand (dealer then plays)

Strategy Tips:
- Always hit on 11 or less (you can't bust)
- Usually stand on 17 or higher
- Consider dealer's showing card when deciding
"""

requirements = """Make ONE decision per turn. Either call hit() OR stand().
After making your decision, explain your reasoning briefly.
Consider the dealer's visible card and your current hand value."""


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BLACKJACK BENCHMARK - FUNCTION TEST")
    print("=" * 60)

    # Test initial deal
    game = create_game(seed=42)

    # Manually deal cards for testing
    game.player_hand.add_card(game.deck.draw())
    game.player_hand.add_card(game.deck.draw())
    game.dealer_hand.add_card(game.deck.draw())
    game.dealer_hand.add_card(game.deck.draw())
    game.phase = GamePhase.PLAYER_TURN

    print("\n[Initial Deal]")
    print(f"  Player: {game.player_hand.describe()}")
    print(f"  Dealer showing: {game.get_visible_dealer_card()}")

    print("\n[Game State]")
    state = get_game_state()
    for k, v in state.items():
        print(f"  {k}: {v}")

    print("\n[Basic Strategy Hint]")
    print(f"  {get_basic_strategy_hint()}")

    print("\n[Test HIT]")
    result = hit()
    print(f"  {result}")

    # Simulate drawing a card
    new_card = game.deck.draw()
    game.player_hand.add_card(new_card)
    print(f"  Drew: {new_card}")
    print(f"  New hand: {game.player_hand.describe()}")

    print("\n[Test STAND]")
    game.player_action = None  # Reset for test
    result = stand()
    print(f"  {result}")

    print("\n" + "=" * 60)
