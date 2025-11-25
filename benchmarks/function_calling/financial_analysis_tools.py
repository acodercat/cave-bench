from typing import Dict, Any, List


def get_stock_prices(symbols: List[str]) -> Dict[str, Any]:
    """
    Get current stock prices for given symbols.
    
    Parameters:
        symbols (List[str]): [Required] List of stock symbols (e.g., ['AAPL', 'TSLA'])
        
    Returns:
        dict: {
            "timestamp": str,           # ISO datetime string
            "prices": {                 # Dictionary of stock prices
                "SYMBOL": {             # Each symbol has:
                    "price": float,     # Current price
                    "change": float     # Price change from previous close
                }
            }
        }
    """
    # Fixed mock stock data - no randomness
    fixed_prices = {
        "AAPL": {"price": 185.50, "change": 2.30},
        "TSLA": {"price": 162.75, "change": -1.85},
        "GOOGL": {"price": 2750.00, "change": 15.20},
        "MSFT": {"price": 420.00, "change": 5.50},
        "AMZN": {"price": 3200.00, "change": -8.75}
    }
    
    result = {
        "timestamp": "2024-04-15T10:30:00",  # Fixed timestamp
        "prices": {}
    }
    
    for symbol in symbols:
        if symbol in fixed_prices:
            result["prices"][symbol] = fixed_prices[symbol]
    
    return result

def calculate_portfolio_value(prices: Dict[str, Any], holdings: Dict[str, int]) -> Dict[str, Any]:
    """
    Calculate total portfolio value based on stock prices and holdings.
    
    Parameters:
        prices (dict): [Required] Stock price data from get_stock_prices(), e.g. {"AAPL": {"price": 185.50, "change": 2.30}}
        holdings (dict): [Required] Number of shares for each symbol, e.g., {"AAPL": 100, "TSLA": 50}
            
    Returns:
        dict: {
            "total_value": float,       # Total portfolio value in USD
            "positions": {              # Dictionary of positions
                "SYMBOL": {             # Each position has:
                    "shares": int,      # Number of shares
                    "value": float,     # Position value (shares * price)
                    "weight": float     # Weight as percentage of total
                }
            }
        }
    """
    positions = {}
    total_value = 0.0
    
    for symbol, shares in holdings.items():
        if symbol in prices:
            price = prices[symbol]["price"]
            position_value = shares * price
            total_value += position_value
            
            positions[symbol] = {
                "shares": shares,
                "value": round(position_value, 2),
                "weight": 0  # Will calculate after total is known
            }
    
    # Calculate weights as percentages
    for symbol in positions:
        positions[symbol]["weight"] = round((positions[symbol]["value"] / total_value) * 100, 2)
    
    return {
        "total_value": round(total_value, 2),
        "positions": positions
    }

def calculate_risk_score(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate risk metrics for the portfolio.
    
    Parameters:
        portfolio_data (dict): [Required] Portfolio data from calculate_portfolio_value(), e.g. {"total_value": 95437.5, "positions": {"AAPL": {"shares": 100, "value": 18550.0, "weight": 19.44}}}
        
    Returns:
        dict: {
            "risk_score": float,               # Risk score from 1-10 (higher = riskier)
            "diversification_score": float,    # Diversification score 0-100 (higher = better)
            "largest_position": {              # Information about largest position
                "symbol": str,                 # Stock symbol
                "weight": float                # Weight as percentage
            }
        }
    """
    positions = portfolio_data["positions"]
    weights = [pos["weight"] for pos in positions.values()]
    
    if not weights:
        return {"risk_score": 10.0, "diversification_score": 0.0, "largest_position": {"symbol": "", "weight": 0}}
    
    max_weight = max(weights)
    num_positions = len(positions)
    
    # Risk score: higher concentration = higher risk
    risk_score = min(10.0, 1.0 + (max_weight / 10.0) + max(0, (5 - num_positions)) * 0.5)
    
    # Diversification: lower concentration = better diversification
    diversification_score = max(0, 100 - max_weight * 1.5)
    
    # Find largest position
    largest_symbol = max(positions.items(), key=lambda x: x[1]["weight"])[0]
    
    return {
        "risk_score": round(risk_score, 2),
        "diversification_score": round(diversification_score, 2),
        "largest_position": {
            "symbol": largest_symbol,
            "weight": positions[largest_symbol]["weight"]
        }
    }

def generate_investment_advice(portfolio_value: float, risk_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate investment advice based on portfolio value and risk.
    
    Parameters:
        portfolio_value (float): [Required] Total portfolio value, e.g. 95437.5
        risk_data (dict): [Required] Risk data from calculate_risk_score(), e.g. {"risk_score": 9.2, "diversification_score": 0, "largest_position": {"symbol": "GOOGL", "weight": 72.04}}
        
    Returns:
        dict: {
            "risk_level": str,              # "Low", "Medium", or "High"
            "recommendations": List[str],    # List of recommendation strings
            "action_needed": bool           # Whether immediate action is needed
        }
    """
    risk_score = risk_data["risk_score"]
    diversification = risk_data["diversification_score"]
    largest_position = risk_data["largest_position"]
    
    # Determine risk level
    if risk_score <= 3:
        risk_level = "Low"
    elif risk_score <= 6:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    recommendations = []
    action_needed = False
    
    if diversification < 50:
        recommendations.append("Consider diversifying your portfolio across more stocks")
        action_needed = True
    
    if largest_position["weight"] > 40:
        recommendations.append(f"Your {largest_position['symbol']} position is too large at {largest_position['weight']}%")
        action_needed = True
    
    if portfolio_value < 10000:
        recommendations.append("Consider investing in index funds for better diversification")
    
    if not recommendations:
        recommendations.append("Your portfolio looks well-balanced")
    
    return {
        "risk_level": risk_level,
        "recommendations": recommendations,
        "action_needed": action_needed
    }


tools = [get_stock_prices, calculate_portfolio_value, calculate_risk_score, generate_investment_advice]