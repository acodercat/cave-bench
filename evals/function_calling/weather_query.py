import pandas as pd
from typing import Any


def get_weather(city: str, date: str) -> dict[str, Any]:
    """
    Get weather data for a specific city and date.

    Parameters:
        city (str): [Required] City name (e.g., 'London', 'New York', 'Tokyo')
        date (str): [Required] Date in YYYY-MM-DD format (e.g., '2023-04-28')
    
    Returns:
        dict: {
            "city": str,
            "date": str,
            "data": {
                "temp": float,
                "condition": str,
                "rain_chance": int
            }
        }
    
    Raises:
        ValueError: If weather data is invalid or missing required fields
    """
    # Simulated weather data
    weather_data = {
        "London": {
            "2023-04-28": {"temp": 15, "condition": "Cloudy", "rain_chance": 40},
            "2023-04-29": {"temp": 17, "condition": "Partly Cloudy", "rain_chance": 20},
            "2023-04-30": {"temp": 14, "condition": "Rain", "rain_chance": 80},
        },
        "New York": {
            "2023-04-28": {"temp": 18, "condition": "Sunny", "rain_chance": 10},
            "2023-04-29": {"temp": 22, "condition": "Clear", "rain_chance": 5},
            "2023-04-30": {"temp": 20, "condition": "Partly Cloudy", "rain_chance": 30},
        },
        "Tokyo": {
            "2023-04-28": {"temp": 21, "condition": "Clear", "rain_chance": 5},
            "2023-04-29": {"temp": 23, "condition": "Sunny", "rain_chance": 0},
            "2023-04-30": {"temp": 22, "condition": "Cloudy", "rain_chance": 40},
        }
    }
    
    if city not in weather_data:
        raise ValueError(f"No weather data found for {city}")
    
    if date not in weather_data[city]:
        raise ValueError(f"No weather data found for {city} on {date}")
    
    return {
        "city": city,
        "date": date,
        "data": weather_data[city][date]
    }

def get_weather_recommendation(weather):
    """
    Provide clothing and activity recommendations based on weather data.
    
    Parameters:
        weather (dict): [Required] Weather data dictionary containing:
            - city (str): [Required] City name
            - date (str): [Required] Date in YYYY-MM-DD format
            - data (dict): [Required] Weather data with:
                - temp (float): [Required] Temperature in Celsius
                - condition (str): [Required] Weather condition
                - rain_chance (int): [Required] Chance of rain (0-100)
    
    Returns:
        dict: {
            "city": str,
            "date": str,
            "recommendations": {
                "clothing": str,
                "umbrella": bool,
                "outdoor_activity": str
            }
        }
    
    Raises:
        ValueError: If weather data is invalid or missing required fields
    """
    if not isinstance(weather, dict) or "data" not in weather:
        raise ValueError("Invalid weather data format")
    
    weather_data = weather["data"]
    required_fields = ["temp", "condition", "rain_chance"]
    if not all(field in weather_data for field in required_fields):
        raise ValueError(f"Missing required weather fields: {required_fields}")
    
    temp = weather_data["temp"]
    condition = weather_data["condition"]
    rain_chance = weather_data["rain_chance"]
    
    return pd.DataFrame({
        "city": weather["city"],
        "date": weather["date"],
        "recommendations": {
            "clothing": get_clothing_recommendation(temp, rain_chance),
            "umbrella": rain_chance > 50,
            "outdoor_activity": get_activity_recommendation(temp, condition, rain_chance)
        }
    })

def get_clothing_recommendation(temperature, rain_chance):
    """Helper function: Get clothing recommendations"""
    if temperature < 10:
        return "Winter clothing, heavy coat required"
    elif temperature < 15:
        return "Fall clothing, light jacket or sweater recommended"
    elif temperature < 22:
        return "Spring clothing, long-sleeve shirt or light sweater"
    else:
        return "Summer clothing, t-shirt appropriate"

def get_activity_recommendation(temperature, condition, rain_chance):
    """Helper function: Get activity recommendations"""
    if rain_chance > 70 or "Rain" in condition:
        return "Outdoor activities not recommended, consider indoor activities like movies or shopping"
    elif temperature > 30:
        return "Hot weather, consider water activities or air-conditioned venues"
    elif temperature < 5:
        return "Cold weather, indoor activities recommended or bundle up for brief outdoor activities"
    else:
        return "Suitable for outdoor activities like park walks, cycling, or picnics"


tools = [get_weather, get_weather_recommendation]


    