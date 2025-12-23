"""
Financial Multi-Turn Analysis Tools

Tests multi-turn conversations with function chaining:
- Sequential dependencies (output of one → input of another)
- Parallel calls within same turn
- State management across turns
"""

from typing import Dict, Any, List

# Simulated account state
_account_state = {
    "balance": 100000.00,
    "watchlist": ["NVDA", "META"],
    "holdings": {}
}

# Fixed stock prices for deterministic tests
_STOCK_PRICES = {
    "AAPL": {"price": 185.50, "change": 2.30, "name": "Apple Inc."},
    "TSLA": {"price": 162.75, "change": -1.85, "name": "Tesla Inc."},
    "GOOGL": {"price": 2750.00, "change": 15.20, "name": "Alphabet Inc."},
    "MSFT": {"price": 420.00, "change": 5.50, "name": "Microsoft Corp."},
    "AMZN": {"price": 3200.00, "change": -8.75, "name": "Amazon.com Inc."},
    "NVDA": {"price": 875.00, "change": 12.50, "name": "NVIDIA Corp."},
    "META": {"price": 505.00, "change": 3.25, "name": "Meta Platforms Inc."}
}

# Cache for chaining function results
_last_prices = {}
_last_portfolio = {}
_last_risk = {}


def get_stock_prices(symbols: List[str]) -> Dict[str, Any]:
    """
    Get current stock prices for given symbols.

    Parameters:
        symbols (List[str]): [Required] List of stock symbols (e.g., ['AAPL', 'TSLA'])

    Returns:
        dict: Dictionary of stock prices, e.g. {"AAPL": {"price": 185.50, "change": 2.30}}
    """
    global _last_prices

    result = {}
    for symbol in symbols:
        if symbol in _STOCK_PRICES:
            result[symbol] = {
                "price": _STOCK_PRICES[symbol]["price"],
                "change": _STOCK_PRICES[symbol]["change"]
            }

    _last_prices = result
    return result


def calculate_portfolio_value(holdings: Dict[str, int]) -> Dict[str, Any]:
    """
    Calculate total portfolio value based on holdings.
    Uses prices from previous get_stock_prices call or fetches them.

    Parameters:
        holdings (dict): [Required] Number of shares for each symbol, e.g., {"AAPL": 100, "TSLA": 50}

    Returns:
        dict: {
            "total_value": float,
            "positions": {
                "SYMBOL": {"shares": int, "value": float, "weight": float}
            }
        }
    """
    global _last_portfolio, _last_prices

    # Use cached prices or fetch new ones
    prices = _last_prices if _last_prices else {s: {"price": _STOCK_PRICES[s]["price"]} for s in holdings if s in _STOCK_PRICES}

    positions = {}
    total_value = 0.0

    for symbol, shares in holdings.items():
        if symbol in _STOCK_PRICES:
            price = prices.get(symbol, {}).get("price", _STOCK_PRICES[symbol]["price"])
            position_value = shares * price
            total_value += position_value
            positions[symbol] = {
                "shares": shares,
                "value": round(position_value, 2),
                "weight": 0
            }

    # Calculate weights
    if total_value > 0:
        for symbol in positions:
            positions[symbol]["weight"] = round((positions[symbol]["value"] / total_value) * 100, 2)

    result = {
        "total_value": round(total_value, 2),
        "positions": positions
    }

    _last_portfolio = result
    return result


def calculate_risk_score(portfolio_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Calculate risk metrics for the portfolio.
    Uses data from previous calculate_portfolio_value call if not provided.

    Parameters:
        portfolio_data (dict): [Optional] Portfolio data. If not provided, uses last calculated portfolio.

    Returns:
        dict: {
            "risk_score": float (1-10),
            "risk_level": str,
            "diversification_score": float (0-100),
            "concentration_warning": bool,
            "largest_position": {"symbol": str, "weight": float}
        }
    """
    global _last_risk, _last_portfolio

    data = portfolio_data if portfolio_data else _last_portfolio

    if not data or "positions" not in data:
        return {"risk_score": 10.0, "risk_level": "Unknown", "diversification_score": 0,
                "concentration_warning": True, "largest_position": {"symbol": "", "weight": 0}}

    positions = data["positions"]
    weights = [pos["weight"] for pos in positions.values()]

    if not weights:
        return {"risk_score": 10.0, "risk_level": "Unknown", "diversification_score": 0,
                "concentration_warning": True, "largest_position": {"symbol": "", "weight": 0}}

    max_weight = max(weights)
    num_positions = len(positions)

    # Risk calculation
    risk_score = min(10.0, 1.0 + (max_weight / 10.0) + max(0, (5 - num_positions)) * 0.5)
    risk_score = round(risk_score, 2)

    # Risk level
    if risk_score <= 3:
        risk_level = "Low"
    elif risk_score <= 6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Diversification score
    diversification_score = round(max(0, 100 - max_weight * 1.5), 2)

    # Concentration warning
    concentration_warning = max_weight > 50

    # Largest position
    largest_symbol = max(positions.items(), key=lambda x: x[1]["weight"])[0]

    result = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "diversification_score": diversification_score,
        "concentration_warning": concentration_warning,
        "largest_position": {
            "symbol": largest_symbol,
            "weight": positions[largest_symbol]["weight"]
        }
    }

    _last_risk = result
    return result


def generate_investment_advice(portfolio_value: float = None, risk_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate investment advice based on portfolio and risk analysis.
    Uses data from previous calls if not provided.

    Parameters:
        portfolio_value (float): [Optional] Total portfolio value. Uses cached if not provided.
        risk_data (dict): [Optional] Risk data. Uses cached if not provided.

    Returns:
        dict: {
            "risk_assessment": str,
            "recommendations": List[str],
            "action_priority": str,
            "suggested_actions": List[str]
        }
    """
    global _last_portfolio, _last_risk

    pv = portfolio_value if portfolio_value else _last_portfolio.get("total_value", 0)
    rd = risk_data if risk_data else _last_risk

    recommendations = []
    suggested_actions = []

    risk_score = rd.get("risk_score", 5)
    diversification = rd.get("diversification_score", 50)
    concentration_warning = rd.get("concentration_warning", False)
    largest = rd.get("largest_position", {})

    # Risk assessment
    if risk_score <= 3:
        risk_assessment = "Your portfolio has low risk with good diversification."
    elif risk_score <= 6:
        risk_assessment = "Your portfolio has moderate risk. Consider minor adjustments."
    else:
        risk_assessment = "Your portfolio has high risk. Immediate attention recommended."

    # Recommendations
    if concentration_warning:
        recommendations.append(f"WARNING: {largest.get('symbol', 'Unknown')} represents {largest.get('weight', 0)}% of your portfolio")
        suggested_actions.append(f"Consider reducing {largest.get('symbol', 'Unknown')} position by 20-30%")

    if diversification < 40:
        recommendations.append("Portfolio diversification is poor")
        suggested_actions.append("Add 2-3 stocks from different sectors")

    if pv < 25000:
        recommendations.append("Portfolio size is relatively small")
        suggested_actions.append("Consider dollar-cost averaging to build position")

    if not recommendations:
        recommendations.append("Portfolio is well-balanced")
        suggested_actions.append("Continue monitoring quarterly")

    # Action priority
    if risk_score > 7 or concentration_warning:
        action_priority = "High"
    elif risk_score > 4:
        action_priority = "Medium"
    else:
        action_priority = "Low"

    return {
        "risk_assessment": risk_assessment,
        "recommendations": recommendations,
        "action_priority": action_priority,
        "suggested_actions": suggested_actions
    }


def get_account_balance() -> Dict[str, Any]:
    """
    Get current account balance and buying power.

    Returns:
        dict: {
            "cash_balance": float,
            "buying_power": float,
            "pending_orders": int
        }
    """
    return {
        "cash_balance": _account_state["balance"],
        "buying_power": _account_state["balance"] * 2,  # 2x margin
        "pending_orders": 0
    }


def get_watchlist() -> Dict[str, Any]:
    """
    Get user's stock watchlist.

    Returns:
        dict: {
            "symbols": List[str],
            "count": int
        }
    """
    return {
        "symbols": _account_state["watchlist"].copy(),
        "count": len(_account_state["watchlist"])
    }


def add_to_watchlist(symbols: List[str]) -> Dict[str, Any]:
    """
    Add stocks to watchlist.

    Parameters:
        symbols (List[str]): [Required] Stock symbols to add, e.g., ["AAPL", "MSFT"]

    Returns:
        dict: {
            "added": List[str],
            "already_exists": List[str],
            "new_watchlist": List[str]
        }
    """
    added = []
    already_exists = []

    for symbol in symbols:
        if symbol in _account_state["watchlist"]:
            already_exists.append(symbol)
        else:
            _account_state["watchlist"].append(symbol)
            added.append(symbol)

    return {
        "added": added,
        "already_exists": already_exists,
        "new_watchlist": _account_state["watchlist"].copy()
    }


def simulate_trade(symbol: str, shares: int, action: str) -> Dict[str, Any]:
    """
    Simulate a buy or sell trade (does not execute, just calculates).

    Parameters:
        symbol (str): [Required] Stock symbol, e.g., "AAPL"
        shares (int): [Required] Number of shares to trade
        action (str): [Required] "buy" or "sell"

    Returns:
        dict: {
            "symbol": str,
            "action": str,
            "shares": int,
            "price_per_share": float,
            "total_cost": float,
            "estimated_fees": float,
            "can_execute": bool
        }
    """
    if symbol not in _STOCK_PRICES:
        return {"error": f"Unknown symbol: {symbol}"}

    price = _STOCK_PRICES[symbol]["price"]
    total_cost = shares * price
    fees = round(total_cost * 0.001, 2)  # 0.1% fee

    if action == "buy":
        can_execute = _account_state["balance"] >= (total_cost + fees)
    else:
        current_shares = _account_state["holdings"].get(symbol, 0)
        can_execute = current_shares >= shares

    return {
        "symbol": symbol,
        "action": action,
        "shares": shares,
        "price_per_share": price,
        "total_cost": round(total_cost, 2),
        "estimated_fees": fees,
        "can_execute": can_execute
    }


def reset_state():
    """Reset all state for testing."""
    global _account_state, _last_prices, _last_portfolio, _last_risk
    _account_state = {
        "balance": 100000.00,
        "watchlist": ["NVDA", "META"],
        "holdings": {}
    }
    _last_prices = {}
    _last_portfolio = {}
    _last_risk = {}


# Export tools
tools = [
    get_stock_prices,
    calculate_portfolio_value,
    calculate_risk_score,
    generate_investment_advice,
    get_account_balance,
    get_watchlist,
    add_to_watchlist,
    simulate_trade
]
