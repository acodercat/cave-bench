"""
Order Batch Processing Tools

Demonstrates CaveAgent's advantage: single-turn execution of loops + conditionals
for processing multiple orders with different statuses and actions.

JSON function calling requires: get_orders → for each order: check status → take action
CaveAgent: single loop with if/elif/else branching
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta


# Fixed "current" date for reproducible tests
_CURRENT_DATE = datetime(2024, 6, 15)

# Simulated order data - deterministic for reproducible tests
_ORDERS_DATA = {
    "ORD001": {
        "order_id": "ORD001",
        "customer": "Alice Smith",
        "status": "paid",
        "amount": 150.00,
        "created_date": "2024-06-14",  # 1 day ago
        "items": ["Laptop Stand", "USB Cable"]
    },
    "ORD002": {
        "order_id": "ORD002",
        "customer": "Bob Johnson",
        "status": "paid",
        "amount": 89.99,
        "created_date": "2024-06-13",  # 2 days ago
        "items": ["Wireless Mouse"]
    },
    "ORD003": {
        "order_id": "ORD003",
        "customer": "Carol White",
        "status": "cancelled",
        "amount": 299.00,
        "created_date": "2024-06-12",  # 3 days ago
        "items": ["Mechanical Keyboard"]
    },
    "ORD004": {
        "order_id": "ORD004",
        "customer": "David Brown",
        "status": "cancelled",
        "amount": 45.50,
        "created_date": "2024-06-10",  # 5 days ago
        "items": ["Phone Case", "Screen Protector"]
    },
    "ORD005": {
        "order_id": "ORD005",
        "customer": "Eve Davis",
        "status": "pending",
        "amount": 199.99,
        "created_date": "2024-06-10",  # 5 days ago - should get reminder (> 3 days)
        "items": ["Bluetooth Headphones"]
    },
    "ORD006": {
        "order_id": "ORD006",
        "customer": "Frank Miller",
        "status": "pending",
        "amount": 75.00,
        "created_date": "2024-06-14",  # 1 day ago - no reminder needed (< 3 days)
        "items": ["Webcam"]
    },
    "ORD007": {
        "order_id": "ORD007",
        "customer": "Grace Wilson",
        "status": "pending",
        "amount": 520.00,
        "created_date": "2024-06-05",  # 10 days ago - should get reminder
        "items": ["Monitor", "HDMI Cable"]
    },
    "ORD008": {
        "order_id": "ORD008",
        "customer": "Henry Taylor",
        "status": "shipped",
        "amount": 35.00,
        "created_date": "2024-06-11",  # Already shipped - no action
        "items": ["USB Hub"]
    }
}

# Track actions for validation
_actions_log = []


def get_pending_orders() -> List[Dict[str, Any]]:
    """
    Get all orders that need processing (excludes already shipped/completed).

    Returns:
        List[dict]: List of orders, each containing:
            - order_id (str): Unique order identifier
            - customer (str): Customer name
            - status (str): Order status - "paid", "cancelled", "pending", or "shipped"
            - amount (float): Order total amount
            - created_date (str): Date order was created (YYYY-MM-DD)
            - items (List[str]): List of items in the order

    Example:
        [
            {"order_id": "ORD001", "customer": "Alice", "status": "paid", ...},
            {"order_id": "ORD002", "customer": "Bob", "status": "cancelled", ...}
        ]
    """
    return list(_ORDERS_DATA.values())


def get_order_details(order_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific order.

    Parameters:
        order_id (str): The order ID (e.g., "ORD001")

    Returns:
        dict: Order details including:
            - order_id (str): Unique order identifier
            - customer (str): Customer name
            - status (str): Order status
            - amount (float): Order total
            - created_date (str): Creation date
            - items (List[str]): Items in order
            - days_since_created (int): Number of days since order was created

    Raises:
        ValueError: If order_id is not found
    """
    if order_id not in _ORDERS_DATA:
        raise ValueError(f"Order '{order_id}' not found")

    order = _ORDERS_DATA[order_id].copy()
    created = datetime.strptime(order["created_date"], "%Y-%m-%d")
    order["days_since_created"] = (_CURRENT_DATE - created).days

    return order


def ship_order(order_id: str) -> Dict[str, Any]:
    """
    Ship an order (mark as shipped and initiate shipping process).

    Parameters:
        order_id (str): The order ID to ship

    Returns:
        dict: {
            "order_id": str,
            "action": "shipped",
            "success": bool,
            "tracking_number": str (if successful)
        }
    """
    if order_id not in _ORDERS_DATA:
        return {"order_id": order_id, "action": "shipped", "success": False, "error": "Order not found"}

    order = _ORDERS_DATA[order_id]
    if order["status"] != "paid":
        return {"order_id": order_id, "action": "shipped", "success": False, "error": f"Cannot ship order with status '{order['status']}'"}

    _actions_log.append({"order_id": order_id, "action": "shipped"})

    return {
        "order_id": order_id,
        "action": "shipped",
        "success": True,
        "tracking_number": f"TRK{order_id[3:]}"
    }


def refund_order(order_id: str) -> Dict[str, Any]:
    """
    Process refund for a cancelled order.

    Parameters:
        order_id (str): The order ID to refund

    Returns:
        dict: {
            "order_id": str,
            "action": "refunded",
            "success": bool,
            "refund_amount": float (if successful)
        }
    """
    if order_id not in _ORDERS_DATA:
        return {"order_id": order_id, "action": "refunded", "success": False, "error": "Order not found"}

    order = _ORDERS_DATA[order_id]
    if order["status"] != "cancelled":
        return {"order_id": order_id, "action": "refunded", "success": False, "error": f"Cannot refund order with status '{order['status']}'"}

    _actions_log.append({"order_id": order_id, "action": "refunded", "amount": order["amount"]})

    return {
        "order_id": order_id,
        "action": "refunded",
        "success": True,
        "refund_amount": order["amount"]
    }


def send_reminder(order_id: str) -> Dict[str, Any]:
    """
    Send a payment reminder for a pending order.

    Parameters:
        order_id (str): The order ID to send reminder for

    Returns:
        dict: {
            "order_id": str,
            "action": "reminder_sent",
            "success": bool,
            "customer": str (if successful)
        }
    """
    if order_id not in _ORDERS_DATA:
        return {"order_id": order_id, "action": "reminder_sent", "success": False, "error": "Order not found"}

    order = _ORDERS_DATA[order_id]
    if order["status"] != "pending":
        return {"order_id": order_id, "action": "reminder_sent", "success": False, "error": f"Cannot send reminder for order with status '{order['status']}'"}

    _actions_log.append({"order_id": order_id, "action": "reminder_sent", "customer": order["customer"]})

    return {
        "order_id": order_id,
        "action": "reminder_sent",
        "success": True,
        "customer": order["customer"]
    }


def get_current_date() -> str:
    """
    Get the current date (for calculating order age).

    Returns:
        str: Current date in YYYY-MM-DD format
    """
    return _CURRENT_DATE.strftime("%Y-%m-%d")


def get_actions_log() -> List[Dict[str, Any]]:
    """
    Get log of all actions taken (for testing/validation).

    Returns:
        List[dict]: List of actions taken
    """
    return _actions_log.copy()


def clear_actions_log() -> None:
    """Clear the actions log (for testing)."""
    global _actions_log
    _actions_log = []


# Export tools for the benchmark
tools = [
    get_pending_orders,
    get_order_details,
    ship_order,
    refund_order,
    send_reminder,
    get_current_date
]
