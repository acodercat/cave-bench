from typing import Dict, Any, List


def get_customer_info(customer_id: str) -> Dict[str, Any]:
    """
    Get customer information by ID.
    
    Parameters:
        customer_id (str): [Required] Customer ID like "CUST123"
        
    Returns:
        dict: {
            "customer_id": str,        # Customer ID
            "name": str,               # Customer full name
            "email": str,              # Customer email
            "tier": str,               # Customer tier: "Gold", "Silver", "Bronze"
            "address": {               # Shipping address
                "street": str,         # Street address
                "city": str,           # City name
                "state": str,          # State code
                "zip": str             # ZIP code
            }
        }
    """
    # Fixed customer database - no randomness
    customers = {
        "CUST123": {
            "customer_id": "CUST123",
            "name": "John Smith",
            "email": "john@email.com",
            "tier": "Gold",
            "address": {
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "zip": "10001"
            }
        },
        "CUST456": {
            "customer_id": "CUST456",
            "name": "Sarah Johnson",
            "email": "sarah@email.com",
            "tier": "Silver",
            "address": {
                "street": "456 Oak Ave",
                "city": "Los Angeles",
                "state": "CA",
                "zip": "90210"
            }
        }
    }
    
    if customer_id not in customers:
        return {"error": f"Customer {customer_id} not found"}
    
    return customers[customer_id]

def check_inventory(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check if items are available in inventory.
    
    Parameters:
        items (List[dict]): [Required] List of items to check
            Each item: {"product": str, "quantity": int}
            
    Returns:
        dict: {
            "all_available": bool,     # True if all items available
            "items": {                 # Inventory check for each item
                "PRODUCT_NAME": {      # For each product:
                    "available": bool, # Whether item is available
                    "in_stock": int,   # Current stock level
                    "requested": int   # Requested quantity
                }
            }
        }
    """
    # Fixed inventory levels - no randomness
    inventory = {
        "iPhone 15": {"stock": 50},
        "AirPods Pro": {"stock": 100},
        "MacBook Pro": {"stock": 25},
        "iPad": {"stock": 75}
    }
    
    all_available = True
    items_check = {}
    
    for item in items:
        product = item["product"]
        quantity = item["quantity"]
        
        if product in inventory:
            in_stock = inventory[product]["stock"]
            available = in_stock >= quantity
            
            items_check[product] = {
                "available": available,
                "in_stock": in_stock,
                "requested": quantity
            }
            
            if not available:
                all_available = False
        else:
            items_check[product] = {
                "available": False,
                "in_stock": 0,
                "requested": quantity
            }
            all_available = False
    
    return {
        "all_available": all_available,
        "items": items_check
    }

def calculate_total(items: List[Dict[str, Any]], customer_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate order total with discounts and tax.
    
    Parameters:
        items (List[dict]): [Required] Items to purchase
            Each item: {"product": str, "quantity": int}
        customer_info (dict): [Required] Customer info from get_customer_info(), e.g. {"customer_id": "CUST123", "name": "John Smith", "email": "john@email.com", "tier": "Gold", "address": {"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}}
        
    Returns:
        dict: {
            "subtotal": float,         # Subtotal before discounts
            "discount": float,         # Discount amount
            "tax": float,              # Tax amount
            "total": float,            # Final total
            "breakdown": List[dict]    # Per-item breakdown
                # Each item: {"product": str, "quantity": int, "unit_price": float, "total": float}
        }
    """
    # Fixed product prices - no randomness
    prices = {
        "iPhone 15": 799.00,
        "AirPods Pro": 249.00,
        "MacBook Pro": 1999.00,
        "iPad": 329.00
    }
    
    # Fixed tier discounts
    tier_discounts = {"Gold": 0.10, "Silver": 0.05, "Bronze": 0.02}
    
    breakdown = []
    subtotal = 0
    
    for item in items:
        product = item["product"]
        quantity = item["quantity"]
        unit_price = prices.get(product, 0)
        item_total = unit_price * quantity
        
        breakdown.append({
            "product": product,
            "quantity": quantity,
            "unit_price": unit_price,
            "total": item_total
        })
        
        subtotal += item_total
    
    # Apply discount
    discount_rate = tier_discounts.get(customer_info.get("tier", "Bronze"), 0)
    discount = subtotal * discount_rate
    
    # Calculate tax (8% for this example)
    tax = (subtotal - discount) * 0.08
    
    total = subtotal - discount + tax
    
    return {
        "subtotal": round(subtotal, 2),
        "discount": round(discount, 2),
        "tax": round(tax, 2),
        "total": round(total, 2),
        "breakdown": breakdown
    }

def process_payment(customer_info: Dict[str, Any], total_amount: float) -> Dict[str, Any]:
    """
    Process payment for the order.
    
    Parameters:
        customer_info (dict): [Required] Customer information, e.g. {"customer_id": "CUST123", "name": "John Smith", "email": "john@email.com", "tier": "Gold", "address": {"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}}
        total_amount (float): [Required] Amount to charge
        
    Returns:
        dict: {
            "success": bool,           # Whether payment succeeded
            "transaction_id": str,     # Transaction ID if successful
            "amount": float,           # Amount charged
            "payment_method": str      # Payment method used
        }
    """
    # Fixed payment processing - always succeeds with predictable transaction ID
    # Generate deterministic transaction ID based on customer and amount
    customer_id = customer_info.get("customer_id", "UNKNOWN")
    transaction_id = f"TXN{customer_id[-3:]}{int(total_amount):04d}"
    
    return {
        "success": True,
        "transaction_id": transaction_id,
        "amount": total_amount,
        "payment_method": "Credit Card"
    }

def create_shipping_label(customer_info: Dict[str, Any], items: List[Dict[str, Any]], payment_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create shipping label for successful orders.
    
    Parameters:
        customer_info (dict): [Required] Customer information, e.g. {"customer_id": "CUST123", "name": "John Smith", "email": "john@email.com", "tier": "Gold", "address": {"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}}
        items (List[dict]): [Required] Items to ship, e.g. [{"product": "iPhone 15", "quantity": 1}, {"product": "AirPods Pro", "quantity": 2}]
        payment_result (dict): [Required] Payment processing result, e.g. {"success": true, "transaction_id": "TXN456789", "amount": 1796.14, "payment_method": "Credit Card"}
        
    Returns:
        dict: {
            "tracking_number": str,    # Shipping tracking number
            "shipping_method": str,    # Shipping method
            "estimated_delivery": str, # Estimated delivery date YYYY-MM-DD
            "shipping_cost": float,    # Shipping cost
            "carrier": str             # Shipping carrier
        }
    """
    if not payment_result.get("success", False):
        return {"error": "Cannot create shipping label - payment failed"}
    
    # Fixed shipping logic based on location - no randomness
    state = customer_info["address"]["state"]
    customer_id = customer_info.get("customer_id", "UNKNOWN")
    
    # Generate deterministic tracking number
    tracking_number = f"TRK{customer_id[-3:]}{state}001"
    
    if state in ["NY", "NJ", "CT"]:
        delivery_days = 1
        shipping_method = "Next Day"
        shipping_cost = 15.99
        estimated_delivery = "2024-04-16"  # Fixed date
    elif state in ["CA", "TX", "FL"]:
        delivery_days = 2
        shipping_method = "2-Day"
        shipping_cost = 9.99
        estimated_delivery = "2024-04-17"  # Fixed date
    else:
        delivery_days = 3
        shipping_method = "Ground"
        shipping_cost = 7.99
        estimated_delivery = "2024-04-18"  # Fixed date
    
    return {
        "tracking_number": tracking_number,
        "shipping_method": shipping_method,
        "estimated_delivery": estimated_delivery,
        "shipping_cost": shipping_cost,
        "carrier": "FedEx"
    }

tools = [get_customer_info, check_inventory, calculate_total, process_payment, create_shipping_label]