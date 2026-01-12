"""
Evening Home Routine - Multi-Turn Benchmark

Tests CaveAgent's ability to:
- Maintain state across conversation turns
- Perform conditional logic based on current device states
- Execute multi-step control routines
- Handle state dependencies between turns
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
from .types import Light, Thermostat, Lock, Camera, Speaker


# ============================================================================
# INITIAL DEVICE STATE (Simulating "just got home from work")
# ============================================================================

# Lights - some were left on when leaving in the morning
living_room_light = Light("Living Room Light", "Living Room", is_on=False, brightness=0)
bedroom_light = Light("Bedroom Light", "Bedroom", is_on=False, brightness=0)
kitchen_light = Light("Kitchen Light", "Kitchen", is_on=True, brightness=60)  # Left on!
hallway_light = Light("Hallway Light", "Hallway", is_on=True, brightness=40)  # Left on!

# Thermostat - house got cold while away
thermostat = Thermostat("Main Thermostat", "Hallway", current_temp=18, target_temp=18)

# Doors - back door was left unlocked (oops!)
front_door_lock = Lock("Front Door", "Entrance", is_locked=True)
back_door_lock = Lock("Back Door", "Back Entrance", is_locked=False)  # Unlocked!

# Camera - on but not recording
security_camera = Camera("Security Camera", "Front Door", is_on=True)
security_camera.is_recording = False

# Speaker - off
living_room_speaker = Speaker("Living Room Speaker", "Living Room", is_on=False)


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_arrival_check(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 1: Validate arrival security check.
    
    OUTCOME-BASED VALIDATION:
    - All doors must be locked
    - Camera must be recording
    
    We don't care HOW the agent discovered which lights were on,
    only that it performed the security actions correctly.
    """
    try:
        front_lock = runtime.retrieve("front_door_lock")
        back_lock = runtime.retrieve("back_door_lock")
        camera = runtime.retrieve("security_camera")
        
        errors = []
        
        # Outcome 1: All doors must be locked
        if not front_lock.is_locked:
            errors.append("Front door should be locked")
        if not back_lock.is_locked:
            errors.append("Back door should be locked")
        
        # Outcome 2: Camera must be recording
        if not camera.is_recording:
            errors.append("Security camera should be recording")
        
        if not errors:
            return ValidatorResult(True, 
                "Arrival check complete! All doors locked, camera recording.")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


def validate_evening_setup(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 2: Validate evening comfort setup.
    
    OUTCOME-BASED VALIDATION:
    - Thermostat target should be 22°C (because initial 18°C < 21°C)
    - Living room light should be ON at 70%
    - Living room speaker should be PLAYING at 40% volume
    """
    try:
        thermostat = runtime.retrieve("thermostat")
        lr_light = runtime.retrieve("living_room_light")
        lr_speaker = runtime.retrieve("living_room_speaker")
        
        errors = []
        
        # Outcome 1: Thermostat set correctly (conditional logic test)
        # Initial current_temp was 18°C, which is < 21°C, so should set to 22°C
        if thermostat.target_temp != 22:
            errors.append(
                f"Thermostat should be 22°C (initial temp 18°C < 21°C threshold), "
                f"got {thermostat.target_temp}°C"
            )
        
        # Outcome 2: Living room light at 70%
        if not lr_light.is_on:
            errors.append("Living room light should be ON")
        elif lr_light.brightness != 70:
            errors.append(f"Living room light should be 70%, got {lr_light.brightness}%")
        
        # Outcome 3: Speaker playing at 40%
        if not lr_speaker.is_playing:
            errors.append("Living room speaker should be playing")
        if lr_speaker.volume != 40:
            errors.append(f"Speaker volume should be 40%, got {lr_speaker.volume}%")
        
        if not errors:
            return ValidatorResult(True, 
                f"Evening setup complete! Thermostat: {thermostat.target_temp}°C, "
                f"Light: {lr_light.brightness}%, Speaker: {lr_speaker.volume}%")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


def validate_bedtime_routine(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 3: Validate bedtime wind-down.
    
    OUTCOME-BASED VALIDATION:
    - Living room light: OFF
    - Living room speaker: OFF
    - Bedroom light: ON at 15%
    - Thermostat: 19°C (because Turn 2 set it to 22°C, which is > 20°C)
    - All doors: LOCKED
    - Camera: RECORDING
    
    KEY TEST: The thermostat check depends on state from Turn 2.
    This validates cross-turn state persistence.
    """
    try:
        lr_light = runtime.retrieve("living_room_light")
        lr_speaker = runtime.retrieve("living_room_speaker")
        bedroom_light = runtime.retrieve("bedroom_light")
        thermostat = runtime.retrieve("thermostat")
        front_lock = runtime.retrieve("front_door_lock")
        back_lock = runtime.retrieve("back_door_lock")
        camera = runtime.retrieve("security_camera")
        
        errors = []
        
        # Outcome 1: Living room OFF
        if lr_light.is_on:
            errors.append("Living room light should be OFF")
        if lr_speaker.is_on or lr_speaker.is_playing:
            errors.append("Living room speaker should be OFF")
        
        # Outcome 2: Bedroom light at 15%
        if not bedroom_light.is_on:
            errors.append("Bedroom light should be ON")
        elif bedroom_light.brightness != 15:
            errors.append(f"Bedroom light should be 15%, got {bedroom_light.brightness}%")
        
        # Outcome 3: Thermostat lowered (CROSS-TURN STATE TEST)
        # Turn 2 set target to 22°C. Since 22 > 20, should lower to 19°C.
        if thermostat.target_temp != 19:
            errors.append(
                f"Thermostat should be 19°C (Turn 2 set it to 22°C > 20°C threshold), "
                f"got {thermostat.target_temp}°C"
            )
        
        # Outcome 4: Security confirmed
        if not front_lock.is_locked:
            errors.append("Front door should be locked")
        if not back_lock.is_locked:
            errors.append("Back door should be locked")
        if not camera.is_recording:
            errors.append("Camera should be recording")
        
        if not errors:
            return ValidatorResult(True, 
                f"Bedtime complete! LR off, bedroom {bedroom_light.brightness}%, "
                f"thermostat {thermostat.target_temp}°C, security OK")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []  # No function tools - object manipulation only

variables = [
    Variable("living_room_light", living_room_light, "Smart light in living room"),
    Variable("bedroom_light", bedroom_light, "Smart light in bedroom"),
    Variable("kitchen_light", kitchen_light, "Smart light in kitchen"),
    Variable("hallway_light", hallway_light, "Smart light in hallway"),
    Variable("living_room_speaker", living_room_speaker, "Smart speaker in living room"),
    Variable(
        "thermostat", 
        thermostat, 
        "Main thermostat"
    ),
    Variable("front_door_lock", front_door_lock, "Front door smart lock"),
    Variable("back_door_lock", back_door_lock, "Back door smart lock"),
    Variable("security_camera", security_camera, "Security camera at front door"),
]

types = [
    Type(Light),
    Type(Thermostat),
    Type(Lock),
    Type(Camera),
    Type(Speaker),
]

validators = {
    "validate_arrival_check": validate_arrival_check,
    "validate_evening_setup": validate_evening_setup,
    "validate_bedtime_routine": validate_bedtime_routine,
}