"""
Movie Night with Baby - Constraint-Based Multi-Turn Benchmark

Tests:
- Temperature range validation
- Entertainment setup with volume constraints
- Power calculation
- End-of-night security routine
"""

from typing import List
from cave_agent import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import Turn, ToolCall
from .types import Light, Thermostat, Lock, Blinds, CoffeeMaker, RobotVacuum, GarageDoor, TemperatureSensor


# Sensors
outdoor_sensor = TemperatureSensor("Outdoor Sensor", "Outside", temperature=14.5, humidity=55)
living_room_sensor = TemperatureSensor("Living Room Sensor", "Living Room", temperature=19.0, humidity=65)

# Climate
thermostat = Thermostat("Main Thermostat", "Hallway", current_temp=19, target_temp=19)

# Lights
bedroom_light = Light("Bedroom Light", "Bedroom", is_on=False, brightness=0)
kitchen_light = Light("Kitchen Light", "Kitchen", is_on=False, brightness=0)
living_room_light = Light("Living Room Light", "Living Room", is_on=False, brightness=0)

# Blinds
bedroom_blinds = Blinds("Bedroom Blinds", "Bedroom", position=0)  # Closed overnight
living_room_blinds = Blinds("Living Room Blinds", "Living Room", position=0)
kitchen_blinds = Blinds("Kitchen Blinds", "Kitchen", position=0)

# Coffee maker - low on resources
coffee_maker = CoffeeMaker("Coffee Maker", "Kitchen", water_level=35, beans_level=25)

# Robot vacuum - ready to clean
robot_vacuum = RobotVacuum("Robot Vacuum", "Living Room", battery_level=85, dustbin_level=30)

# Security
front_door_lock = Lock("Front Door", "Entrance", is_locked=True)
garage_door = GarageDoor("Garage Door", "Garage", is_open=True)  # Left open

# ============================================================================
# VALIDATORS
# ============================================================================

def validate_morning_climate_check(
    response: str,
    runtime: PythonRuntime,
    turn: Turn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 1: Climate assessment.
    
    Initial state:
    - outdoor_sensor.temperature = 14.5°C (< 18°C, so COLD)
    - living_room_sensor.humidity = 65% (> 60%, so HIGH)
    
    Expected:
    - thermostat.target_temp = 23 (cold morning)
    """
    try:
        thermostat = runtime.retrieve("thermostat")
        
        errors = []
        
        # Cold morning (14.5°C < 18°C) → should set to 23°C
        if thermostat.target_temp != 23:
            errors.append(
                f"Thermostat should be 23°C (outdoor temp 14.5°C < 18°C threshold), "
                f"got {thermostat.target_temp}°C"
            )
        

        
        if not errors:
            return ValidatorResult(True, 
                f"Morning climate check complete! Thermostat: {thermostat.target_temp}°C")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


def validate_morning_prep(
    response: str,
    runtime: PythonRuntime,
    turn: Turn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 2: Morning preparation.
    
    Initial state:
    - bedroom_blinds.position = 0 (closed)
    - coffee_maker: water=35%, beans=25% (both sufficient)
    - kitchen_light: off
    
    Expected:
    - bedroom_blinds.position = 100 (open)
    - coffee_maker.is_brewing = True or cups_ready = 2
    - kitchen_light: on at 60%
    """
    try:
        blinds = runtime.retrieve("bedroom_blinds")
        coffee = runtime.retrieve("coffee_maker")
        kitchen = runtime.retrieve("kitchen_light")
        
        errors = []
        
        # Blinds should be open
        if blinds.position != 100:
            errors.append(f"Bedroom blinds should be fully open (100), got {blinds.position}")
        
        # Coffee maker should have brewed (water=35 >= 20, beans=25 >= 10)
        # After brewing 2 cups: water -= 20, beans -= 10
        if not coffee.is_brewing and coffee.cups_ready != 2:
            errors.append(
                f"Coffee maker should be brewing/have 2 cups ready "
                f"(had sufficient resources), got cups_ready={coffee.cups_ready}"
            )
        
        # Kitchen light at 60%
        if not kitchen.is_on:
            errors.append("Kitchen light should be ON")
        elif kitchen.brightness != 60:
            errors.append(f"Kitchen light should be 60%, got {kitchen.brightness}%")
        
        if not errors:
            return ValidatorResult(True, "Morning prep complete!")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


def validate_leaving_for_work(
    response: str,
    runtime: PythonRuntime,
    turn: Turn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 3: Leaving for work automation.
    
    Initial state (from setup):
    - robot_vacuum: battery=85% (>50%), dustbin=30% (<80%) → should clean
    - garage_door: is_open=True → should close
    
    Expected:
    - All blinds closed (position=0)
    - robot_vacuum cleaning living room in quiet mode
    - coffee_maker off
    - kitchen_light off
    - thermostat: mode=eco, target=18°C
    - front_door locked
    - garage_door closed
    """
    try:
        bedroom_blinds = runtime.retrieve("bedroom_blinds")
        living_room_blinds = runtime.retrieve("living_room_blinds")
        kitchen_blinds = runtime.retrieve("kitchen_blinds")
        vacuum = runtime.retrieve("robot_vacuum")
        coffee = runtime.retrieve("coffee_maker")
        kitchen_light = runtime.retrieve("kitchen_light")
        thermostat = runtime.retrieve("thermostat")
        front_lock = runtime.retrieve("front_door_lock")
        garage = runtime.retrieve("garage_door")
        
        errors = []
        
        # All blinds closed
        if bedroom_blinds.position != 0:
            errors.append(f"Bedroom blinds should be closed (0), got {bedroom_blinds.position}")
        if living_room_blinds.position != 0:
            errors.append(f"Living room blinds should be closed (0), got {living_room_blinds.position}")
        if kitchen_blinds.position != 0:
            errors.append(f"Kitchen blinds should be closed (0), got {kitchen_blinds.position}")
        
        # Robot vacuum should be cleaning (battery 85% > 50%, dustbin 30% < 80%)
        if not vacuum.is_cleaning:
            errors.append(
                "Robot vacuum should be cleaning (battery=85%>50%, dustbin=30%<80%)"
            )
        elif vacuum.current_room != "living room" and vacuum.current_room != "Living Room":
            errors.append(f"Vacuum should be cleaning living room, got {vacuum.current_room}")
        elif vacuum.cleaning_mode != "quiet":
            errors.append(f"Vacuum should be in quiet mode, got {vacuum.cleaning_mode}")
        
        # Coffee maker off
        if coffee.is_on:
            errors.append("Coffee maker should be OFF")
        
        # Kitchen light off
        if kitchen_light.is_on:
            errors.append("Kitchen light should be OFF")
        
        # Thermostat eco mode at 18°C
        if thermostat.mode != "eco":
            errors.append(f"Thermostat should be in eco mode, got {thermostat.mode}")
        if thermostat.target_temp != 18:
            errors.append(f"Thermostat should be 18°C, got {thermostat.target_temp}°C")
        
        # Front door locked
        if not front_lock.is_locked:
            errors.append("Front door should be locked")
        
        # Garage door closed
        if garage.is_open:
            errors.append("Garage door should be closed")
        
        if not errors:
            return ValidatorResult(True, "Left for work - home secured!")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable(
        "outdoor_sensor", outdoor_sensor,
        "Outdoor temp/humidity sensor"
    ),
    Variable(
        "living_room_sensor", living_room_sensor,
        "Living room temp/humidity sensor"
    ),
    Variable(
        "thermostat", thermostat,
        "Thermostat"
    ),
    Variable(
        "bedroom_light", bedroom_light,
        "Bedroom light"
    ),
    Variable(
        "kitchen_light", kitchen_light,
        "Kitchen light"
    ),
    Variable(
        "living_room_light", living_room_light,
        "Living room light"
    ),
    Variable(
        "bedroom_blinds", bedroom_blinds,
        "Bedroom blinds"
    ),
    Variable(
        "living_room_blinds", living_room_blinds,
        "Living room blinds"
    ),
    Variable(
        "kitchen_blinds", kitchen_blinds,
        "Kitchen blinds"
    ),
    Variable(
        "coffee_maker", coffee_maker,
        "Coffee maker"
    ),
    Variable(
        "robot_vacuum", robot_vacuum,
        "Robot vacuum"
    ),
    Variable(
        "front_door_lock", front_door_lock,
        "Front door lock"
    ),
    Variable(
        "garage_door", garage_door,
        "Garage door"
    ),
]

types = [
    Type(Light),
    Type(Thermostat),
    Type(Lock),
    Type(Blinds),
    Type(TemperatureSensor),
    Type(CoffeeMaker),
    Type(RobotVacuum),
    Type(GarageDoor),
]

validators = {
    "validate_morning_climate_check": validate_morning_climate_check,
    "validate_morning_prep": validate_morning_prep,
    "validate_leaving_for_work": validate_leaving_for_work,
}