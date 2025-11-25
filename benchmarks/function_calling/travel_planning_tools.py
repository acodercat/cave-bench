from typing import Dict, Any, List


def search_flights(origin: str, destination: str, date: str) -> Dict[str, Any]:
    """
    Search for flights between two cities.
    
    Parameters:
        origin (str): [Required] Origin city code (e.g., 'NYC')
        destination (str): [Required] Destination city code (e.g., 'LAX')
        date (str): [Required] Travel date in YYYY-MM-DD format
        
    Returns:
        dict: {
            "flight_number": str,    # Flight number
            "airline": str,          # Airline name
            "departure_time": str,   # Departure time HH:MM
            "arrival_time": str,     # Arrival time HH:MM
            "price": float,          # Ticket price in USD
            "duration": str          # Flight duration like "5h 30m"
        }
    """
    # Fixed flight data based on route - no randomness
    route_data = {
        ("NYC", "LAX"): {
            "flight_number": "AA256",
            "airline": "American",
            "departure_time": "08:30",
            "arrival_time": "11:45",
            "price": 387.50,
            "duration": "5h 15m"
        },
        ("LAX", "NYC"): {
            "flight_number": "AA257",
            "airline": "American", 
            "departure_time": "14:20",
            "arrival_time": "22:35",
            "price": 412.00,
            "duration": "5h 15m"
        },
        ("NYC", "MIA"): {
            "flight_number": "DL445",
            "airline": "Delta",
            "departure_time": "09:15",
            "arrival_time": "12:30",
            "price": 298.75,
            "duration": "3h 15m"
        },
        ("CHI", "LAX"): {
            "flight_number": "UA512",
            "airline": "United",
            "departure_time": "10:00",
            "arrival_time": "12:20",
            "price": 342.80,
            "duration": "4h 20m"
        }
    }
    
    route = (origin, destination)
    if route in route_data:
        return route_data[route]
    else:
        # Default flight for unknown routes
        return {
            "flight_number": "SW789",
            "airline": "Southwest",
            "departure_time": "12:00",
            "arrival_time": "16:30",
            "price": 400.00,
            "duration": "4h 30m"
        }

def get_weather_forecast(city: str, date: str) -> Dict[str, Any]:
    """
    Get weather forecast for a specific city and date.
    
    Parameters:
        city (str): [Required] City name (e.g., 'Los Angeles')
        date (str): [Required] Date in YYYY-MM-DD format
        
    Returns:
        dict: {
            "city": str,           # City name
            "date": str,           # Date in YYYY-MM-DD format
            "temperature": int,     # Temperature in Fahrenheit
            "condition": str,       # Weather condition like "Sunny", "Rainy"
            "rain_chance": int      # Chance of rain as percentage 0-100
        }
    """
    # Fixed weather data by city and date - no randomness
    weather_data = {
        ("Los Angeles", "2024-06-15"): {
            "temperature": 78,
            "condition": "Sunny",
            "rain_chance": 15
        },
        ("Los Angeles", "2024-07-20"): {
            "temperature": 82,
            "condition": "Partly Cloudy", 
            "rain_chance": 20
        },
        ("New York", "2024-06-15"): {
            "temperature": 68,
            "condition": "Cloudy",
            "rain_chance": 45
        },
        ("Miami", "2024-06-15"): {
            "temperature": 84,
            "condition": "Partly Cloudy",
            "rain_chance": 30
        },
        ("Chicago", "2024-07-20"): {
            "temperature": 75,
            "condition": "Sunny",
            "rain_chance": 10
        }
    }
    
    key = (city, date)
    if key in weather_data:
        weather = weather_data[key]
    else:
        # Default weather for unknown city/date combinations
        weather = {
            "temperature": 72,
            "condition": "Partly Cloudy",
            "rain_chance": 25
        }
    
    return {
        "city": city,
        "date": date,
        "temperature": weather["temperature"],
        "condition": weather["condition"],
        "rain_chance": weather["rain_chance"]
    }

def find_hotels(city: str, date: str, flight_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find hotel recommendations based on city, date and flight arrival.
    
    Parameters:
        city (str): [Required] City name
        date (str): [Required] Check-in date in YYYY-MM-DD format
        flight_info (dict): [Required] Flight information from search_flights(), e.g. {"flight_number": "AA256", "airline": "American", "departure_time": "08:30", "arrival_time": "11:45", "price": 387.50, "duration": "5h 15m"}
        
    Returns:
        dict: {
            "hotel_name": str,     # Name of recommended hotel
            "price_per_night": float,  # Price per night in USD
            "rating": float,       # Hotel rating out of 5.0
            "location": str,       # Hotel location/area
            "amenities": List[str] # List of hotel amenities
        }
    """
    # Fixed hotel recommendations by city - no randomness
    hotels_by_city = {
        "Los Angeles": {
            "hotel_name": "Downtown LA Hotel",
            "price_per_night": 150.0,
            "rating": 4.2,
            "location": "Downtown",
            "amenities": ["WiFi", "Pool", "Gym"]
        },
        "New York": {
            "hotel_name": "Manhattan Plaza",
            "price_per_night": 250.0,
            "rating": 4.3,
            "location": "Midtown",
            "amenities": ["WiFi", "Gym", "Restaurant"]
        },
        "Miami": {
            "hotel_name": "South Beach Resort",
            "price_per_night": 220.0,
            "rating": 4.6,
            "location": "South Beach",
            "amenities": ["WiFi", "Pool", "Restaurant", "Room Service"]
        },
        "Chicago": {
            "hotel_name": "Chicago Downtown Hotel",
            "price_per_night": 180.0,
            "rating": 4.1,
            "location": "Downtown",
            "amenities": ["WiFi", "Gym", "Restaurant"]
        }
    }
    
    if city in hotels_by_city:
        return hotels_by_city[city]
    else:
        # Default hotel for unknown cities
        return {
            "hotel_name": "City Hotel",
            "price_per_night": 120.0,
            "rating": 4.0,
            "location": "City Center",
            "amenities": ["WiFi", "Restaurant"]
        }

def suggest_activities(city: str, weather: Dict[str, Any], hotel: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest activities based on weather and hotel location.
    
    Parameters:
        city (str): [Required] City name
        weather (dict): [Required] Weather data from get_weather_forecast(), e.g. {"city": "Los Angeles", "date": "2024-06-15", "temperature": 78, "condition": "Sunny", "rain_chance": 15}
        hotel (dict): [Required] Hotel data from find_hotels(), e.g. {"hotel_name": "Downtown LA Hotel", "price_per_night": 150.0, "rating": 4.2, "location": "Downtown", "amenities": ["WiFi", "Pool", "Gym"]}
        
    Returns:
        dict: {
            "morning_activity": {      # Morning activity suggestion
                "name": str,           # Activity name
                "type": str,           # Activity type like "Outdoor", "Indoor"
                "cost": float          # Estimated cost in USD
            },
            "afternoon_activity": {    # Afternoon activity suggestion
                "name": str,
                "type": str,
                "cost": float
            },
            "weather_note": str        # Note about weather impact
        }
    """
    rain_chance = weather["rain_chance"]
    condition = weather["condition"]
    
    # Fixed activity recommendations by city and weather - no randomness
    outdoor_activities = {
        "Los Angeles": {"name": "Hollywood Walk of Fame", "cost": 0},
        "New York": {"name": "Central Park", "cost": 0},
        "Miami": {"name": "South Beach", "cost": 0},
        "Chicago": {"name": "Millennium Park", "cost": 0}
    }
    
    indoor_activities = {
        "Los Angeles": {"name": "Getty Center", "cost": 20},
        "New York": {"name": "Metropolitan Museum", "cost": 30},
        "Miami": {"name": "Pérez Art Museum", "cost": 20},
        "Chicago": {"name": "Art Institute", "cost": 25}
    }
    
    # Choose activities based on weather
    if rain_chance > 60:
        activity = indoor_activities.get(city, {"name": "Local Museum", "cost": 15})
        activity_type = "Indoor"
        weather_note = f"Indoor activities recommended due to {condition.lower()} weather"
    else:
        activity = outdoor_activities.get(city, {"name": "City Walking Tour", "cost": 20})
        activity_type = "Outdoor"
        weather_note = f"Great weather for outdoor activities - {condition.lower()}"
    
    return {
        "morning_activity": {
            "name": activity["name"],
            "type": activity_type,
            "cost": activity["cost"]
        },
        "afternoon_activity": {
            "name": activity["name"],
            "type": activity_type,
            "cost": activity["cost"]
        },
        "weather_note": weather_note
    }

tools = [search_flights, get_weather_forecast, find_hotels, suggest_activities]