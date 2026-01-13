from typing import Any
def get_available_flights(departure_date: str, origin_airport: str, destination_airport: str) -> list[dict[str, Any]]:
    """
    Get available flights between origin and destination airports on a specific date.
    
    Parameters:
    - departure_date (str): Date in YYYY-MM-DD format
    - origin_airport (str): Three-letter airport code for origin
    - destination_airport (str): Three-letter airport code for destination
    
    Returns:
    - list: List of dictionaries containing flight information, each dictionary contains the following keys:
    {
        "flight_id": "CX234",
            "airline": "Cathay Pacific",
            "departure_time": "08:30",
            "arrival_time": "14:45",
            "duration": 730,
            "price": 3500.0,
            "stops": 0,
            "layover_airports": []
        },
    ]
            
    """
    # Simulated flight data
    return [
        {
            "flight_id": "CX234",
            "airline": "Cathay Pacific",
            "departure_time": "08:30",
            "arrival_time": "14:45",
            "duration": 730,
            "price": 3500.0,
            "stops": 0,
            "layover_airports": []
        },
        {
            "flight_id": "BA28",
            "airline": "British Airways",
            "departure_time": "10:15",
            "arrival_time": "16:30",
            "duration": 750,
            "price": 2800.0,
            "stops": 0,
            "layover_airports": []
        },
        {
            "flight_id": "EK231",
            "airline": "Emirates",
            "departure_time": "14:20",
            "arrival_time": "01:35",
            "duration": 960,
            "price": 2950.0,
            "stops": 1,
            "layover_airports": ["DXB"]
        },
        {
            "flight_id": "QR817",
            "airline": "Qatar Airways",
            "departure_time": "15:45",
            "arrival_time": "23:10",
            "duration": 920,
            "price": 2750.0,
            "stops": 1,
            "layover_airports": ["DOH"]
        },
        {
            "flight_id": "VS201",
            "airline": "Virgin Atlantic",
            "departure_time": "16:30",
            "arrival_time": "22:45",
            "duration": 780,
            "price": 2650.0,
            "stops": 0,
            "layover_airports": []
        },
        {
            "flight_id": "LH797",
            "airline": "Lufthansa",
            "departure_time": "18:20",
            "arrival_time": "06:15",
            "duration": 960,
            "price": 2450.0,
            "stops": 1,
            "layover_airports": ["FRA"]
        }
    ]

def check_seat_availability(flight_id: str, seat_class: str = None) -> dict[str, Any]:
    """
    Check seat availability for a specific flight and class.
    
    Parameters:
    - flight_id (str): Unique identifier for the flight (required).
    - seat_class (str, optional): Class of seat (economy, premium_economy, business, first). If None, returns data for all seat classes.
    
    Returns:
    - dict: Dictionary containing seat availability information. Example return formats:
       {
         "economy": {"available": True, "remaining_seats": 42, "price": 3500.0},
         "premium_economy": {"available": True, "remaining_seats": 12, "price": 4800.0},
         ...
       }
    """
    # Simulated availability data
    availability_data = {
        "CX234": {
            "economy": {"available": True, "remaining_seats": 42, "price": 3500.0},
            "premium_economy": {"available": True, "remaining_seats": 12, "price": 4800.0},
            "business": {"available": True, "remaining_seats": 5, "price": 8500.0},
            "first": {"available": False, "remaining_seats": 0, "price": 12000.0}
        },
        "BA28": {
            "economy": {"available": True, "remaining_seats": 15, "price": 2800.0},
            "premium_economy": {"available": True, "remaining_seats": 8, "price": 3900.0},
            "business": {"available": True, "remaining_seats": 3, "price": 7200.0},
            "first": {"available": True, "remaining_seats": 1, "price": 11500.0}
        },
        "EK231": {
            "economy": {"available": True, "remaining_seats": 35, "price": 2950.0},
            "premium_economy": {"available": True, "remaining_seats": 18, "price": 4100.0},
            "business": {"available": True, "remaining_seats": 7, "price": 7800.0},
            "first": {"available": True, "remaining_seats": 2, "price": 12500.0}
        },
        "QR817": {
            "economy": {"available": True, "remaining_seats": 22, "price": 2750.0},
            "premium_economy": {"available": False, "remaining_seats": 0, "price": 3800.0},
            "business": {"available": True, "remaining_seats": 4, "price": 7500.0},
            "first": {"available": True, "remaining_seats": 1, "price": 13000.0}
        },
        "VS201": {
            "economy": {"available": True, "remaining_seats": 18, "price": 2650.0},
            "premium_economy": {"available": True, "remaining_seats": 6, "price": 3700.0},
            "business": {"available": True, "remaining_seats": 2, "price": 6900.0},
            "first": {"available": False, "remaining_seats": 0, "price": 10500.0}
        },
        "LH797": {
            "economy": {"available": True, "remaining_seats": 28, "price": 2450.0},
            "premium_economy": {"available": True, "remaining_seats": 10, "price": 3400.0},
            "business": {"available": False, "remaining_seats": 0, "price": 6500.0},
            "first": {"available": False, "remaining_seats": 0, "price": 9800.0}
        }
    }
    
    # Case 1: Only flight_id provided - return all seat classes for that flight
    if seat_class is None:
        if flight_id not in availability_data:
            return {}  # Return empty dict if flight not found
        return availability_data[flight_id]
    
    # Case 2: Both flight_id and seat_class provided - return specific seat class for specific flight
    if flight_id in availability_data and seat_class in availability_data[flight_id]:
        return availability_data[flight_id][seat_class]
    else:
        return {"available": False, "remaining_seats": 0, "price": 0.0}

def get_passenger_information_by_id(passenger_id: str) -> dict[str, Any]:
    """
    Get passenger information for a given passenger ID.
    
    Parameters:
    - passenger_id (str): Unique identifier for the passenger
    
    Returns:
    - dict: Dictionary containing passenger information, each key is a passenger ID and the value is a dictionary containing the following keys:
    {
        "name": "John Smith",
        "passport": "AB123456",
        "nationality": "United Kingdom",
        "frequent_flyer": {
            "program": "Avios",
            "number": "AV78923456",
            "tier": "Silver"
        },
        "preferences": {
            "seat": "window",
            "meal": "regular"
        }
    }
    """
    # Simulated passenger data
    passengers = {
        "P12345": {
            "name": "John Smith",
            "passport": "AB123456",
            "nationality": "United Kingdom",
            "frequent_flyer": {
                "program": "Avios",
                "number": "AV78923456",
                "tier": "Silver"
            },
            "preferences": {
                "seat": "window",
                "meal": "regular"
            }
        },
        "P67890": {
            "name": "Emma Chen",
            "passport": "CD789012",
            "nationality": "Hong Kong",
            "frequent_flyer": {
                "program": "Asia Miles",
                "number": "AM45678901",
                "tier": "Gold"
            },
            "preferences": {
                "seat": "aisle",
                "meal": "vegetarian"
            }
        }
    }
    
    return passengers.get(passenger_id, {})

def get_passenger_information_by_name(name: str) -> dict[str, Any]:
    """
    Get passenger information for a given passenger name.
    
    Parameters:
    - name (str): Name of the passenger, ex: "John Smith"
    
    Returns:
    - dict: Dictionary containing passenger information, each key is a passenger ID and the value is a dictionary containing the following keys:
    {
        "passenger_id": "P45321",
        "name": "John Smith",
        "passport": "AB123456",
        "nationality": "United Kingdom",
        "frequent_flyer": {
            "program": "Avios",
            "number": "AV78923456",
            "tier": "Silver"
        },
        "preferences": {
            "seat": "window",
            "meal": "regular"
        }
    }
    """
    # Simulated passenger data
    passengers = {
        "John Smith": {
            "passenger_id": "P12345",
            "name": "John Smith",
            "passport": "AB123456",
            "nationality": "United Kingdom",
            "frequent_flyer": {
                "program": "Avios",
                "number": "AV78923456",
                "tier": "Silver"
            },
            "preferences": {
                "seat": "window",
                "meal": "regular"
            }
        },
        "Emma Chen": {
            "passenger_id": "P67890",
            "name": "Emma Chen",
            "passport": "CD789012",
            "nationality": "Hong Kong",
            "frequent_flyer": {
                "program": "Asia Miles",
                "number": "AM45678901",
                "tier": "Gold"
            },
            "preferences": {
                "seat": "aisle",
                "meal": "vegetarian"
            }
        }
    }
    
    return passengers.get(name, {})

def book_flight(flight_id: str, passenger_id: str, seat_class: str = "economy") -> dict[str, Any]:
    """
    Book a flight for a passenger.
    
    Parameters:
    - flight_id (str): Unique identifier for the flight, ex: "CX234"
    - passenger_id (str): Unique identifier for the passenger, ex: "P12845"
    - seat_class (str): Class of seat (economy, premium_economy, business, first)
    
    Returns:
    - dict: Dictionary containing booking information, each key is a booking ID and the value is a dictionary containing the following keys:
    {
        "booking_id": "BK123456",
        "status": "confirmed",
        "flight_details": {},
        "passenger_details": {},
        "seat_assignment": "",
        "total_price": 0.0
    }
    """
    # Check seat availability
    seat_availability = check_seat_availability(flight_id, seat_class)
    if not seat_availability["available"]:
        return {
            "booking_id": "",
            "status": "failed",
            "reason": f"No {seat_class} seats available for flight {flight_id}",
            "flight_details": {},
            "passenger_details": {},
            "seat_assignment": "",
            "total_price": 0.0
        }
    
    # Get passenger information
    passenger_info = get_passenger_information_by_id(passenger_id)
    if not passenger_info:
        return {
            "booking_id": "",
            "status": "failed",
            "reason": f"Passenger with ID {passenger_id} not found",
            "flight_details": {},
            "passenger_details": {},
            "seat_assignment": "",
            "total_price": 0.0
        }
    
    # Simulated booking ID generation
    import random
    booking_id = f"BK{random.randint(10000, 99999)}"
    
    # Calculate total price including taxes and fees (simulated)
    base_price = seat_availability["price"]
    taxes_and_fees = base_price * 0.15
    total_price = base_price + taxes_and_fees
    
    # Assign a seat (simulated)
    seat_row = random.randint(1, 30)
    seat_column = random.choice(["A", "B", "C", "D", "E", "F"])
    seat_assignment = f"{seat_row}{seat_column}"
    
    # Get flight details
    flights = get_available_flights("2023-04-28", "HKG", "LHR")  # Using tomorrow's date
    flight_details = next((flight for flight in flights if flight["flight_id"] == flight_id), {})
    
    return {
        "booking_id": booking_id,
        "status": "confirmed",
        "flight_details": flight_details,
        "passenger_details": passenger_info,
        "seat_assignment": seat_assignment,
        "total_price": round(total_price, 2)
    }



tools = [get_available_flights, check_seat_availability, get_passenger_information_by_id, get_passenger_information_by_name, book_flight]
