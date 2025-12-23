"""
Smart Building Optimization Tools

Demonstrates CaveAgent's advantage: single-turn execution of loops + conditionals
vs JSON function calling which requires multiple turns for check-then-decide patterns.
"""

from typing import Dict, List, Any


# Simulated building data - deterministic for reproducible tests
_BUILDING_DATA = {
    "lobby": {"temperature": 28, "occupied": True, "lights": 80},
    "office_a": {"temperature": 16, "occupied": True, "lights": 100},
    "office_b": {"temperature": 22, "occupied": True, "lights": 60},
    "conference": {"temperature": 30, "occupied": False, "lights": 0},
    "break_room": {"temperature": 15, "occupied": False, "lights": 20},
    "server_room": {"temperature": 35, "occupied": False, "lights": 10},
}

# Track actions for validation
_actions_log = []


def get_rooms() -> List[str]:
    """
    Get list of all rooms in the building.

    Returns:
        List[str]: List of room IDs
        Example: ["lobby", "office_a", "office_b", "conference", "break_room", "server_room"]
    """
    return list(_BUILDING_DATA.keys())


def get_room_status(room_id: str) -> Dict[str, Any]:
    """
    Get current status of a specific room.

    Parameters:
        room_id (str): Room identifier (e.g., "lobby", "office_a")

    Returns:
        dict: {
            "room_id": str,
            "temperature": int,  # Current temperature in Celsius
            "occupied": bool,    # Whether room is currently occupied
            "lights": int        # Current light level (0-100)
        }

    Raises:
        ValueError: If room_id is not found
    """
    if room_id not in _BUILDING_DATA:
        raise ValueError(f"Room '{room_id}' not found")

    data = _BUILDING_DATA[room_id]
    return {
        "room_id": room_id,
        "temperature": data["temperature"],
        "occupied": data["occupied"],
        "lights": data["lights"]
    }


def set_ac(room_id: str, on: bool) -> Dict[str, Any]:
    """
    Turn air conditioning on or off for a room.

    Parameters:
        room_id (str): Room identifier
        on (bool): True to turn on, False to turn off

    Returns:
        dict: {
            "room_id": str,
            "device": "ac",
            "action": "on" or "off",
            "success": bool
        }
    """
    if room_id not in _BUILDING_DATA:
        return {"room_id": room_id, "device": "ac", "action": "on" if on else "off", "success": False}

    action = "on" if on else "off"
    _actions_log.append({"room_id": room_id, "device": "ac", "action": action})

    return {
        "room_id": room_id,
        "device": "ac",
        "action": action,
        "success": True
    }


def set_heater(room_id: str, on: bool) -> Dict[str, Any]:
    """
    Turn heater on or off for a room.

    Parameters:
        room_id (str): Room identifier
        on (bool): True to turn on, False to turn off

    Returns:
        dict: {
            "room_id": str,
            "device": "heater",
            "action": "on" or "off",
            "success": bool
        }
    """
    if room_id not in _BUILDING_DATA:
        return {"room_id": room_id, "device": "heater", "action": "on" if on else "off", "success": False}

    action = "on" if on else "off"
    _actions_log.append({"room_id": room_id, "device": "heater", "action": action})

    return {
        "room_id": room_id,
        "device": "heater",
        "action": action,
        "success": True
    }


def set_lights(room_id: str, level: int) -> Dict[str, Any]:
    """
    Set light level for a room.

    Parameters:
        room_id (str): Room identifier
        level (int): Light level from 0 (off) to 100 (full brightness)

    Returns:
        dict: {
            "room_id": str,
            "device": "lights",
            "level": int,
            "success": bool
        }
    """
    if room_id not in _BUILDING_DATA:
        return {"room_id": room_id, "device": "lights", "level": level, "success": False}

    level = max(0, min(100, level))  # Clamp to 0-100
    _actions_log.append({"room_id": room_id, "device": "lights", "level": level})

    return {
        "room_id": room_id,
        "device": "lights",
        "level": level,
        "success": True
    }


def turn_off_climate(room_id: str) -> Dict[str, Any]:
    """
    Turn off all climate control (AC and heater) for a room.

    Parameters:
        room_id (str): Room identifier

    Returns:
        dict: {
            "room_id": str,
            "ac": "off",
            "heater": "off",
            "success": bool
        }
    """
    if room_id not in _BUILDING_DATA:
        return {"room_id": room_id, "ac": "off", "heater": "off", "success": False}

    _actions_log.append({"room_id": room_id, "device": "ac", "action": "off"})
    _actions_log.append({"room_id": room_id, "device": "heater", "action": "off"})

    return {
        "room_id": room_id,
        "ac": "off",
        "heater": "off",
        "success": True
    }


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
    get_rooms,
    get_room_status,
    set_ac,
    set_heater,
    set_lights,
    turn_off_climate
]
