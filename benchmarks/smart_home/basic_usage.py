"""
Smart Home Benchmark - Object Manipulation and Automation Logic

This module provides smart home device classes for testing CaveAgent's ability to:
- Manipulate objects and their properties
- Call methods on objects
- Implement automation logic
- Work with class hierarchies and inheritance
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
from .types import Light, Thermostat, Lock, Camera, Speaker, Room


# ============================================================================
# DEVICE INSTANCES (Variables exposed to agent)
# ============================================================================

# Living room devices
living_room_light = Light("Living Room Light", "Living Room", is_on=True, brightness=80)
living_room_speaker = Speaker("Living Room Speaker", "Living Room", is_on=False)

# Bedroom devices
bedroom_light = Light("Bedroom Light", "Bedroom", is_on=False, brightness=0)
bedroom_speaker = Speaker("Bedroom Speaker", "Bedroom", is_on=False)

# Kitchen devices
kitchen_light = Light("Kitchen Light", "Kitchen", is_on=False, brightness=0)

# Hallway/shared devices
hallway_light = Light("Hallway Light", "Hallway", is_on=True, brightness=50)
thermostat = Thermostat("Main Thermostat", "Hallway", current_temp=20, target_temp=20)

# Security devices
front_door_lock = Lock("Front Door Lock", "Front Door", is_locked=True)
back_door_lock = Lock("Back Door Lock", "Back Door", is_locked=False)
security_camera = Camera("Front Camera", "Front Door", is_on=True)


# Rooms
living_room = Room("Living Room")
living_room.add_device(living_room_light)
living_room.add_device(living_room_speaker)

bedroom = Room("Bedroom")
bedroom.add_device(bedroom_light)
bedroom.add_device(bedroom_speaker)

kitchen = Room("Kitchen")
kitchen.add_device(kitchen_light)

all_devices = [
    living_room_light,
    living_room_speaker,
    bedroom_light,
    bedroom_speaker,
    kitchen_light,
    hallway_light,
    thermostat,
    front_door_lock,
    back_door_lock,
    security_camera,
    living_room,
    bedroom,
    kitchen,
]

# ============================================================================
# VALIDATORS
# ============================================================================

def validate_movie_mode(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate movie mode - living room light should be dimmed to 20%."""
    try:
        lr_light = runtime.get_variable("living_room_light")

        # Check brightness is 20%
        if lr_light.brightness == 20:
            return ValidatorResult(True, f"Movie mode set correctly! Light at {lr_light.brightness}%")
        else:
            return ValidatorResult(False,
                f"Light should be at 20% brightness. Current: {lr_light.brightness}%")
    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


def validate_leaving_home(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate leaving home automation - doors locked, lights off, camera recording."""
    try:
        front_lock = runtime.get_variable("front_door_lock")
        back_lock = runtime.get_variable("back_door_lock")
        all_devices = runtime.get_variable("all_devices")
        security_camera = runtime.get_variable("security_camera")

        errors = []

        # Check 1: All doors locked
        if not front_lock.is_locked:
            errors.append("Front door not locked")
        if not back_lock.is_locked:
            errors.append("Back door not locked")

        # Check 2: All lights are off
        lights_on = [d for d in all_devices if isinstance(d, Light) and d.is_on]
        if lights_on:
            light_names = [l.name for l in lights_on]
            errors.append(f"Lights still on: {', '.join(light_names)}")

        # Check 3: Security camera is on and recording
        if not security_camera.is_on:
            errors.append("Security camera is off")
        if not security_camera.is_recording:
            errors.append("Security camera not recording")

        if not errors:
            return ValidatorResult(True, "Leaving home automation complete!")
        else:
            return ValidatorResult(False, f"Issues found: {'; '.join(errors)}")

    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")
    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


def validate_goodnight_routine(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate goodnight routine - temperature calculation, selective lighting, and security."""
    try:
        thermostat = runtime.get_variable("thermostat")
        bedroom_light = runtime.get_variable("bedroom_light")
        all_devices = runtime.get_variable("all_devices")
        front_lock = runtime.get_variable("front_door_lock")
        back_lock = runtime.get_variable("back_door_lock")
        security_camera = runtime.get_variable("security_camera")
        warmup_time = runtime.get_variable("warmup_time_minutes")

        errors = []

        # Check 1: Thermostat set to 22°C
        if thermostat.target_temp != 22:
            errors.append(f"Thermostat should be 22°C, got {thermostat.target_temp}°C")

        # Check 2: Warmup time calculated correctly (22 - 20 = 2°C, at 0.5°C per 10 min = 40 minutes)
        expected_warmup = 40
        if warmup_time != expected_warmup:
            errors.append(f"Warmup time should be {expected_warmup} minutes, got {warmup_time}")

        # Check 3: Only bedroom light is on, others off
        lights_on = [d for d in all_devices if isinstance(d, Light) and d.is_on]
        if len(lights_on) != 1:
            light_names = [l.name for l in lights_on]
            errors.append(f"Only bedroom light should be on. Currently on: {', '.join(light_names)}")
        elif lights_on[0].name != "Bedroom Light":
            errors.append(f"Bedroom light should be on, but {lights_on[0].name} is on instead")

        # Check 4: Bedroom light dimmed to 20%
        if bedroom_light.brightness != 20:
            errors.append(f"Bedroom light should be 20% brightness, got {bedroom_light.brightness}%")

        # Check 5: All doors locked
        if not front_lock.is_locked:
            errors.append("Front door not locked")
        if not back_lock.is_locked:
            errors.append("Back door not locked")

        # Check 6: Security camera recording
        if not security_camera.is_on:
            errors.append("Security camera is off")
        if not security_camera.is_recording:
            errors.append("Security camera not recording")

        if not errors:
            return ValidatorResult(True, f"Goodnight routine complete! Warmup: {warmup_time} minutes")
        else:
            return ValidatorResult(False, f"Issues found: {'; '.join(errors)}")

    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")
    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


# ============================================================================
# EXPORTS
# ============================================================================

warmup_time_minutes = 0

tools = []  # No function tools - object manipulation only
variables = [
    Variable("living_room_light", living_room_light, "Smart light in living room"),
    Variable("bedroom_light", bedroom_light, "Smart light in bedroom"),
    Variable("kitchen_light", kitchen_light, "Smart light in kitchen"),
    Variable("hallway_light", hallway_light, "Smart light in hallway"),
    Variable("living_room_speaker", living_room_speaker, "Smart speaker in living room",),
    Variable("bedroom_speaker", bedroom_speaker, "Smart speaker in bedroom"),
    Variable("thermostat", thermostat, "Main thermostat for home"),
    Variable("front_door_lock", front_door_lock, "Front door smart lock"),
    Variable("back_door_lock", back_door_lock, "Back door smart lock"),
    Variable("security_camera", security_camera, "Security camera at front door"),
    Variable("living_room", living_room, "Living room with devices"),
    Variable("bedroom", bedroom, "Bedroom with devices"),
    Variable("kitchen", kitchen, "Kitchen with devices"),
    Variable("warmup_time_minutes", warmup_time_minutes, "Warmup time in minutes"),
    Variable("all_devices", all_devices, "List of all smart devices"),
]

types = [
    Type(Light),
    Type(Thermostat),
    Type(Lock),
    Type(Camera),
    Type(Speaker),
    Type(Room),
]

validators = {
    "validate_movie_mode": validate_movie_mode,
    "validate_leaving_home": validate_leaving_home,
    "validate_goodnight_routine": validate_goodnight_routine,
}
