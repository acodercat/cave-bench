from typing import Dict, Any, List


def search_recipes(cuisine: str, dietary_restrictions: List[str]) -> Dict[str, Any]:
    """
    Search for recipes based on cuisine and dietary restrictions.
    
    Parameters:
        cuisine (str): [Required] Cuisine type like "Italian", "Mexican", "Asian"
        dietary_restrictions (List[str]): [Required] List like ["vegetarian", "gluten-free"]
        
    Returns:
        dict: {
            "recipes": List[dict],     # List of recipe options
                # Each recipe: {
                #     "name": str,       # Recipe name
                #     "prep_time": int,  # Prep time in minutes
                #     "difficulty": str, # "Easy", "Medium", "Hard"
                #     "serves": int      # Number of servings
                # }
        }
    """
    # Fixed recipe database - no randomness
    recipe_db = {
        "Italian": [
            {"name": "Spaghetti Carbonara", "prep_time": 30, "difficulty": "Medium", "serves": 4},
            {"name": "Margherita Pizza", "prep_time": 45, "difficulty": "Easy", "serves": 2},
            {"name": "Chicken Parmigiana", "prep_time": 60, "difficulty": "Hard", "serves": 4}
        ],
        "Mexican": [
            {"name": "Chicken Tacos", "prep_time": 25, "difficulty": "Easy", "serves": 4},
            {"name": "Beef Enchiladas", "prep_time": 50, "difficulty": "Medium", "serves": 6},
            {"name": "Vegetarian Quesadillas", "prep_time": 20, "difficulty": "Easy", "serves": 2}
        ],
        "Asian": [
            {"name": "Chicken Stir Fry", "prep_time": 20, "difficulty": "Easy", "serves": 4},
            {"name": "Beef Teriyaki", "prep_time": 35, "difficulty": "Medium", "serves": 4},
            {"name": "Vegetable Pad Thai", "prep_time": 30, "difficulty": "Medium", "serves": 3}
        ]
    }
    
    recipes = recipe_db.get(cuisine, [])
    
    # Filter by dietary restrictions - deterministic filtering
    if "vegetarian" in dietary_restrictions:
        recipes = [r for r in recipes if "Vegetarian" in r["name"] or "Vegetable" in r["name"] or "Margherita" in r["name"]]
    
    return {"recipes": recipes}

def get_ingredients(recipe_name: str, servings: int) -> Dict[str, Any]:
    """
    Get ingredients list for a specific recipe and serving size.
    
    Parameters:
        recipe_name (str): [Required] Name of the recipe
        servings (int): [Required] Number of servings needed
        
    Returns:
        dict: {
            "recipe_name": str,        # Recipe name
            "servings": int,           # Number of servings
            "ingredients": List[dict], # List of ingredients
                # Each ingredient: {
                #     "name": str,       # Ingredient name
                #     "amount": str,     # Amount needed like "2 cups"
                #     "category": str    # Category like "Protein", "Vegetable"
                # }
        }
    """
    # Fixed ingredient database - no randomness
    ingredient_db = {
        "Chicken Tacos": [
            {"name": "Chicken breast", "amount": "1 lb", "category": "Protein"},
            {"name": "Tortillas", "amount": "8 pieces", "category": "Grain"},
            {"name": "Lettuce", "amount": "1 head", "category": "Vegetable"},
            {"name": "Tomatoes", "amount": "2 medium", "category": "Vegetable"},
            {"name": "Cheese", "amount": "1 cup", "category": "Dairy"}
        ],
        "Vegetarian Quesadillas": [
            {"name": "Flour tortillas", "amount": "4 large", "category": "Grain"},
            {"name": "Bell peppers", "amount": "2 medium", "category": "Vegetable"},
            {"name": "Onions", "amount": "1 medium", "category": "Vegetable"},
            {"name": "Cheese", "amount": "2 cups", "category": "Dairy"},
            {"name": "Black beans", "amount": "1 can", "category": "Protein"}
        ],
        "Vegetable Pad Thai": [
            {"name": "Rice noodles", "amount": "8 oz", "category": "Grain"},
            {"name": "Bell peppers", "amount": "2 medium", "category": "Vegetable"},
            {"name": "Carrots", "amount": "2 medium", "category": "Vegetable"},
            {"name": "Bean sprouts", "amount": "1 cup", "category": "Vegetable"},
            {"name": "Peanuts", "amount": "0.5 cup", "category": "Protein"}
        ],
        "Spaghetti Carbonara": [
            {"name": "Spaghetti", "amount": "1 lb", "category": "Grain"},
            {"name": "Eggs", "amount": "4 large", "category": "Protein"},
            {"name": "Bacon", "amount": "6 strips", "category": "Protein"},
            {"name": "Parmesan cheese", "amount": "1 cup", "category": "Dairy"},
            {"name": "Black pepper", "amount": "1 tsp", "category": "Spice"}
        ]
    }
    
    base_ingredients = ingredient_db.get(recipe_name, [])
    
    # Scale ingredients based on servings (assuming base recipe serves 4 for tacos, 2 for quesadillas)
    base_servings_map = {
        "Chicken Tacos": 4,
        "Vegetarian Quesadillas": 2,
        "Vegetable Pad Thai": 3,
        "Spaghetti Carbonara": 4
    }
    
    base_servings = base_servings_map.get(recipe_name, 4)
    scale_factor = servings / base_servings
    
    scaled_ingredients = []
    for ingredient in base_ingredients:
        # Simple scaling for known patterns
        amount_parts = ingredient["amount"].split()
        if len(amount_parts) >= 2:
            try:
                number = float(amount_parts[0]) * scale_factor
                unit = " ".join(amount_parts[1:])
                if number == int(number):
                    scaled_amount = f"{int(number)} {unit}"
                else:
                    scaled_amount = f"{number:.1f} {unit}"
            except ValueError:
                # Handle non-numeric amounts like "1 head"
                if scale_factor > 1.5:
                    scaled_amount = f"{int(scale_factor)} {ingredient['amount']}"
                else:
                    scaled_amount = ingredient["amount"]
        else:
            scaled_amount = ingredient["amount"]
        
        scaled_ingredients.append({
            "name": ingredient["name"],
            "amount": scaled_amount,
            "category": ingredient["category"]
        })
    
    return {
        "recipe_name": recipe_name,
        "servings": servings,
        "ingredients": scaled_ingredients
    }

def check_pantry(ingredients: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check which ingredients are available in pantry vs need to buy.
    
    Parameters:
        ingredients (List[dict]): [Required] Ingredients list from get_ingredients(), e.g. [{"name": "Chicken breast", "amount": "1 lb", "category": "Protein"}]
        
    Returns:
        dict: {
            "have_in_pantry": List[dict],      # Ingredients already available
                # Each item: {"name": str, "amount": str, "category": str}
            "need_to_buy": List[dict],         # Ingredients to purchase
                # Each item: {"name": str, "amount": str, "category": str}
            "shopping_list_total": float       # Estimated cost for missing items
        }
    """
    # Fixed pantry contents - what we typically have at home
    pantry_items = [
        "salt", "black pepper", "olive oil", "garlic", "onions",
        "rice", "pasta", "flour", "sugar", "eggs"
    ]
    
    # Fixed prices for ingredients we need to buy
    ingredient_prices = {
        "Chicken breast": 8.99,
        "Flour tortillas": 3.49,
        "Tortillas": 3.49,
        "Lettuce": 2.99,
        "Tomatoes": 4.99,
        "Cheese": 5.99,
        "Bell peppers": 3.99,
        "Carrots": 1.99,
        "Bean sprouts": 2.49,
        "Peanuts": 4.99,
        "Rice noodles": 2.99,
        "Black beans": 1.29,
        "Spaghetti": 1.99,
        "Bacon": 6.99,
        "Parmesan cheese": 7.99
    }
    
    have_in_pantry = []
    need_to_buy = []
    total_cost = 0
    
    for ingredient in ingredients:
        ingredient_name = ingredient["name"]
        
        # Check if we have it in pantry (case-insensitive partial match)
        have_it = any(pantry_item.lower() in ingredient_name.lower() 
                     for pantry_item in pantry_items)
        
        if have_it:
            have_in_pantry.append(ingredient)
        else:
            need_to_buy.append(ingredient)
            # Add estimated cost
            cost = ingredient_prices.get(ingredient_name, 3.99)  # Default $3.99
            total_cost += cost
    
    return {
        "have_in_pantry": have_in_pantry,
        "need_to_buy": need_to_buy,
        "shopping_list_total": round(total_cost, 2)
    }

def create_shopping_list(pantry_check: Dict[str, Any], store_preference: str) -> Dict[str, Any]:
    """
    Create organized shopping list with store information.
    
    Parameters:
        pantry_check (dict): [Required] Pantry check results from check_pantry(), e.g. {"have_in_pantry": [{"name": "Chicken breast", "amount": "1 lb", "category": "Protein"}], "need_to_buy": [{"name": "Flour tortillas", "amount": "4 large", "category": "Grain"}], "shopping_list_total": 19.45}
        store_preference (str): [Required] Preferred store like "Walmart", "Target", "Whole Foods"
        
    Returns:
        dict: {
            "store": str,                      # Store name
            "organized_list": {                # Items organized by category
                "CATEGORY": List[dict]         # Each category contains list of items
                    # Each item: {"name": str, "amount": str, "estimated_price": float}
            },
            "total_items": int,                # Total number of items
            "estimated_total": float,          # Total estimated cost
            "store_location": str              # Store address/location
        }
    """
    # Fixed store locations and pricing
    store_locations = {
        "Walmart": "123 Main St, Anytown USA",
        "Target": "456 Shopping Blvd, Anytown USA", 
        "Whole Foods": "789 Organic Ave, Anytown USA"
    }
    
    # Fixed price multipliers by store
    price_multipliers = {"Walmart": 1.0, "Target": 1.1, "Whole Foods": 1.3}
    multiplier = price_multipliers.get(store_preference, 1.0)
    
    # Fixed base prices
    base_prices = {
        "Chicken breast": 8.99, "Flour tortillas": 3.49, "Tortillas": 3.49,
        "Lettuce": 2.99, "Tomatoes": 4.99, "Cheese": 5.99,
        "Bell peppers": 3.99, "Carrots": 1.99, "Bean sprouts": 2.49,
        "Peanuts": 4.99, "Rice noodles": 2.99, "Black beans": 1.29,
        "Spaghetti": 1.99, "Bacon": 6.99, "Parmesan cheese": 7.99
    }
    
    # Get items we need to buy
    items_to_buy = pantry_check["need_to_buy"]
    
    # Organize by category
    organized_list = {}
    total_items = 0
    
    for item in items_to_buy:
        category = item["category"]
        if category not in organized_list:
            organized_list[category] = []
        
        base_price = base_prices.get(item["name"], 3.99)
        store_price = round(base_price * multiplier, 2)
        
        organized_list[category].append({
            "name": item["name"],
            "amount": item["amount"],
            "estimated_price": store_price
        })
        total_items += 1
    
    estimated_total = pantry_check["shopping_list_total"] * multiplier
    
    return {
        "store": store_preference,
        "organized_list": organized_list,
        "total_items": total_items,
        "estimated_total": round(estimated_total, 2),
        "store_location": store_locations.get(store_preference, "Location not found")
    }

tools = [search_recipes, get_ingredients, check_pantry, create_shopping_list]