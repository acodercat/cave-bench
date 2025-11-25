from typing import Dict, Any


def check_home_status() -> Dict[str, Any]:
    """
    Check current status of smart home devices.
    
    Returns:
        dict: {
            "temperature": int,            # Current temperature in Fahrenheit
            "lights": {                    # Light status by room
                "ROOM_NAME": {             # Each room has:
                    "on": bool,            # Whether lights are on
                    "brightness": int,     # Brightness 0-100
                    "color": str           # Light color
                }
            },
            "security": {                  # Security system status
                "armed": bool,             # Whether system is armed
                "doors_locked": bool,      # Whether doors are locked
                "motion_detected": bool    # Recent motion detection
            },
            "occupancy": bool              # Whether anyone is home
        }
    """
    # Fixed current home status - no randomness
    return {
        "temperature": 72,
        "lights": {
            "living_room": {
                "on": True,
                "brightness": 80,
                "color": "cool_white"
            },
            "kitchen": {
                "on": False,
                "brightness": 0,
                "color": "warm_white"
            },
            "bedroom": {
                "on": False,
                "brightness": 0,
                "color": "warm_white"
            }
        },
        "security": {
            "armed": False,
            "doors_locked": True,
            "motion_detected": True
        },
        "occupancy": True
    }

def get_user_preferences(time_of_day: str) -> Dict[str, Any]:
    """
    Get user preferences for different times of day.
    
    Parameters:
        time_of_day (str): [Required] Time period like "morning", "evening", "night"
        
    Returns:
        dict: {
            "preferred_temperature": int,      # Preferred temperature
            "lighting_preferences": {          # Lighting preferences by room
                "ROOM_NAME": {                 # Each room has:
                    "brightness": int,         # Preferred brightness 0-100
                    "color": str               # Preferred color
                }
            },
            "security_mode": str,              # Security mode like "home", "away", "sleep"
            "music_preference": {              # Music preferences
                "genre": str,                  # Preferred genre
                "volume": int                  # Preferred volume 0-100
            }
        }
    """
    # Fixed preferences by time of day - no randomness
    preferences = {
        "morning": {
            "preferred_temperature": 72,
            "lighting_preferences": {
                "living_room": {"brightness": 80, "color": "cool_white"},
                "kitchen": {"brightness": 100, "color": "cool_white"},
                "bedroom": {"brightness": 60, "color": "warm_white"}
            },
            "security_mode": "home",
            "music_preference": {"genre": "upbeat", "volume": 60}
        },
        "evening": {
            "preferred_temperature": 70,
            "lighting_preferences": {
                "living_room": {"brightness": 60, "color": "warm_white"},
                "kitchen": {"brightness": 80, "color": "warm_white"},
                "bedroom": {"brightness": 30, "color": "warm_white"}
            },
            "security_mode": "home",
            "music_preference": {"genre": "relaxing", "volume": 40}
        },
        "night": {
            "preferred_temperature": 68,
            "lighting_preferences": {
                "living_room": {"brightness": 20, "color": "warm_white"},
                "kitchen": {"brightness": 30, "color": "warm_white"},
                "bedroom": {"brightness": 10, "color": "red"}
            },
            "security_mode": "sleep",
            "music_preference": {"genre": "ambient", "volume": 20}
        }
    }
    
    return preferences.get(time_of_day, preferences["evening"])

def adjust_thermostat(current_temp: int, preferred_temp: int, occupancy: bool) -> Dict[str, Any]:
    """
    Adjust thermostat based on preferences and occupancy.
    
    Parameters:
        current_temp (int): [Required] Current temperature
        preferred_temp (int): [Required] Preferred temperature
        occupancy (bool): [Required] Whether anyone is home
        
    Returns:
        dict: {
            "action_taken": str,               # Description of action
            "new_target_temp": int,            # New target temperature
            "estimated_time_to_reach": int,    # Minutes to reach target
            "energy_mode": str                 # Energy mode like "normal", "eco"
        }
    """
    if not occupancy:
        # Energy saving mode when no one is home
        if preferred_temp > current_temp:
            new_target = preferred_temp - 3  # Lower heating when away
        else:
            new_target = preferred_temp + 3  # Higher cooling when away
        energy_mode = "eco"
        action = f"Set to eco mode: {new_target}°F (no one home)"
    else:
        new_target = preferred_temp
        energy_mode = "normal"
        if current_temp == preferred_temp:
            action = "No adjustment needed - already at preferred temperature"
        elif current_temp < preferred_temp:
            action = f"Heating to {preferred_temp}°F"
        else:
            action = f"Cooling to {preferred_temp}°F"
    
    # Fixed time calculation (1 degree per 10 minutes)
    temp_difference = abs(current_temp - new_target)
    estimated_time = temp_difference * 10
    
    return {
        "action_taken": action,
        "new_target_temp": new_target,
        "estimated_time_to_reach": estimated_time,
        "energy_mode": energy_mode
    }

def control_lighting(current_lights: Dict[str, Any], lighting_preferences: Dict[str, Any], occupancy: bool) -> Dict[str, Any]:
    """
    Control smart lights based on lighting preferences and occupancy.
    
    Parameters:
        current_lights (dict): [Required] Current light status from check_home_status(), e.g. {"temperature": 72, "lights": {"living_room": {"on": true, "brightness": 80, "color": "cool_white"}, "kitchen": {"on": false, "brightness": 0, "color": "warm_white"}, "bedroom": {"on": false, "brightness": 0, "color": "warm_white"}}, "security": {"armed": false, "doors_locked": true, "motion_detected": true}, "occupancy": true}
        lighting_preferences (dict): [Required] Lighting preferences from get_user_preferences(), e.g. {"living_room": {"brightness": 60, "color": "warm_white"}, "kitchen": {"brightness": 80, "color": "warm_white"}, "bedroom": {"brightness": 30, "color": "warm_white"}}
        occupancy (bool): [Required] Whether anyone is home
        
    Returns:
        dict: {
            "changes_made": List[dict],        # List of changes made
                # Each change: {"room": str, "action": str, "brightness": int, "color": str}
            "energy_saved": bool,              # Whether energy was saved
            "total_rooms_adjusted": int        # Number of rooms adjusted
        }
    """
    changes_made = []
    total_rooms_adjusted = 0
    energy_saved = False
    
    if not occupancy:
        # Turn off all lights when no one is home
        for room in current_lights:
            if current_lights[room]["on"]:
                changes_made.append({
                    "room": room,
                    "action": "turned_off",
                    "brightness": 0,
                    "color": current_lights[room]["color"]
                })
                total_rooms_adjusted += 1
                energy_saved = True
    else:
        # Adjust lights based on lighting preferences
        
        for room in current_lights:
            current_room = current_lights[room]
            if room in lighting_preferences:
                pref_room = lighting_preferences[room]
                
                # Check if changes needed
                brightness_change = pref_room["brightness"] != current_room["brightness"]
                color_change = pref_room["color"] != current_room["color"]
                on_change = (pref_room["brightness"] > 0) != current_room["on"]
                
                if brightness_change or color_change or on_change:
                    action = "adjusted"
                    if pref_room["brightness"] == 0:
                        action = "turned_off"
                    elif not current_room["on"]:
                        action = "turned_on"
                    
                    changes_made.append({
                        "room": room,
                        "action": action,
                        "brightness": pref_room["brightness"],
                        "color": pref_room["color"]
                    })
                    total_rooms_adjusted += 1
                    
                    # Check if energy was saved
                    if pref_room["brightness"] < current_room["brightness"]:
                        energy_saved = True
    
    return {
        "changes_made": changes_made,
        "energy_saved": energy_saved,
        "total_rooms_adjusted": total_rooms_adjusted
    }

def set_security_mode(current_security: Dict[str, Any], preferred_mode: str, occupancy: bool) -> Dict[str, Any]:
    """
    Set security system mode based on preferences and occupancy.
    
    Parameters:
        current_security (dict): [Required] Current security status, e.g. {"armed": false, "doors_locked": true, "motion_detected": true}
        preferred_mode (str): [Required] Preferred security mode like "home", "away", "sleep"
        occupancy (bool): [Required] Whether anyone is home
        
    Returns:
        dict: {
            "mode_set": str,                   # Security mode that was set
            "actions_taken": List[str],        # List of security actions taken
            "all_secure": bool                 # Whether home is fully secured
        }
    """
    actions_taken = []
    
    # Override preference if no one is home
    if not occupancy and preferred_mode != "away":
        mode_to_set = "away"
        actions_taken.append("Override to 'away' mode - no one home")
    else:
        mode_to_set = preferred_mode
    
    # Take actions based on mode
    if mode_to_set == "away":
        if not current_security["armed"]:
            actions_taken.append("Armed security system")
        if not current_security["doors_locked"]:
            actions_taken.append("Locked all doors")
        actions_taken.append("Set motion detection to high sensitivity")
        all_secure = True
        
    elif mode_to_set == "sleep":
        if not current_security["doors_locked"]:
            actions_taken.append("Locked all doors")
        actions_taken.append("Armed perimeter sensors only")
        actions_taken.append("Set motion detection to low sensitivity")
        all_secure = True
        
    elif mode_to_set == "home":
        if current_security["armed"]:
            actions_taken.append("Disarmed security system")
        actions_taken.append("Set motion detection to normal sensitivity")
        all_secure = current_security["doors_locked"]
        
    else:
        actions_taken.append("Unknown security mode - no changes made")
        all_secure = False
    
    if not actions_taken:
        actions_taken.append("No security changes needed")
    
    return {
        "mode_set": mode_to_set,
        "actions_taken": actions_taken,
        "all_secure": all_secure
    }


tools = [check_home_status, get_user_preferences, adjust_thermostat, control_lighting, set_security_mode]