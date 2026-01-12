"""
Movie Night with Baby - Constraint-Based Multi-Turn Benchmark

Tests:
- Temperature range validation
- Entertainment setup with volume constraints
- Power calculation
- End-of-night security routine
- Building status reports
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
from .types import Light, Thermostat, Lock, Camera, Speaker, Blinds, TV, Doorbell, TemperatureSensor

# Nursery
nursery_sensor = TemperatureSensor("Nursery Sensor", "Nursery", temperature=19.0, humidity=50)
nursery_camera = Camera("Nursery Camera", "Nursery", is_on=True)
nursery_camera.is_recording = False

# Living room
living_room_light = Light("Living Room Light", "Living Room", is_on=True, brightness=80)
living_room_blinds = Blinds("Living Room Blinds", "Living Room", position=50)
living_room_tv = TV("Living Room TV", "Living Room", is_on=False)
living_room_speaker = Speaker("Living Room Speaker", "Living Room", is_on=False)

# Other lights
kitchen_light = Light("Kitchen Light", "Kitchen", is_on=True, brightness=60)
hallway_light = Light("Hallway Light", "Hallway", is_on=True, brightness=40)

# Climate
thermostat = Thermostat("Main Thermostat", "Hallway", current_temp=19, target_temp=19)

# Security
front_door_lock = Lock("Front Door", "Entrance", is_locked=True)
back_door_lock = Lock("Back Door", "Back Entrance", is_locked=False)  # Unlocked!
doorbell = Doorbell("Front Doorbell", "Front Door")



# ============================================================================
# VALIDATORS
# ============================================================================

def validate_baby_safety_setup(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 1: Baby safety setup.
    
    Initial state:
    - nursery_sensor.temperature = 19.0°C (NOT in 20-22 range)
    - nursery_camera.is_recording = False
    - doorbell.do_not_disturb = False
    
    Expected:
    - thermostat.target_temp = 21 (because 19 not in 20-22)
    - nursery_camera.is_recording = True
    - doorbell.do_not_disturb = True
    """
    try:
        thermostat = runtime.retrieve("thermostat")
        camera = runtime.retrieve("nursery_camera")
        doorbell = runtime.retrieve("doorbell")
        
        errors = []
        
        # Thermostat should be 21 (nursery was 19°C, outside 20-22 range)
        if thermostat.target_temp != 21:
            errors.append(
                f"Thermostat should be 21°C (nursery was 19°C, outside 20-22 range), "
                f"got {thermostat.target_temp}°C"
            )
        
        # Nursery camera recording
        if not camera.is_recording:
            errors.append("Nursery camera should be recording")
        
        # Doorbell DND enabled
        if not doorbell.do_not_disturb:
            errors.append("Doorbell should have do-not-disturb enabled")
        
        if not errors:
            return ValidatorResult(True, "Baby safety setup complete!")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


def validate_movie_setup(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 2: Movie setup.
    
    Expected:
    - TV on, input = 'Netflix'
    - living_room_light: 15% brightness
    - living_room_blinds: position = 0
    - speaker: volume = 35%, playing
    """
    try:
        tv = runtime.retrieve("living_room_tv")
        light = runtime.retrieve("living_room_light")
        blinds = runtime.retrieve("living_room_blinds")
        speaker = runtime.retrieve("living_room_speaker")
        
        errors = []
        
        # TV on with Netflix
        if not tv.is_on:
            errors.append("TV should be ON")
        elif tv.input_source != "Netflix":
            errors.append(f"TV input should be 'Netflix', got '{tv.input_source}'")
        
        # Light at 15%
        if not light.is_on:
            errors.append("Living room light should be ON (dimmed)")
        elif light.brightness != 15:
            errors.append(f"Living room light should be 15%, got {light.brightness}%")
        
        # Blinds closed
        if blinds.position != 0:
            errors.append(f"Blinds should be closed (0), got {blinds.position}")
        
        # Speaker at 35% and playing
        if not speaker.is_playing:
            errors.append("Speaker should be playing")
        if speaker.volume != 35:
            errors.append(f"Speaker volume should be 35%, got {speaker.volume}%")
        
        if not errors:
            return ValidatorResult(True, f"Movie setup complete!")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


def validate_movie_night_end(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 3: End of movie night.
    
    Expected:
    - TV off
    - Speaker off
    - All lights off
    - Nursery camera still recording
    - Doorbell still in DND
    - All doors locked
    """
    try:
        tv = runtime.retrieve("living_room_tv")
        speaker = runtime.retrieve("living_room_speaker")
        lr_light = runtime.retrieve("living_room_light")
        kitchen_light = runtime.retrieve("kitchen_light")
        hallway_light = runtime.retrieve("hallway_light")
        nursery_camera = runtime.retrieve("nursery_camera")
        doorbell = runtime.retrieve("doorbell")
        front_lock = runtime.retrieve("front_door_lock")
        back_lock = runtime.retrieve("back_door_lock")
        
        errors = []
        
        # Entertainment off
        if tv.is_on:
            errors.append("TV should be OFF")
        if speaker.is_on or speaker.is_playing:
            errors.append("Speaker should be OFF")
        
        # All lights off
        all_lights_off = not any([lr_light.is_on, kitchen_light.is_on, hallway_light.is_on])
        if lr_light.is_on:
            errors.append("Living room light should be OFF")
        if kitchen_light.is_on:
            errors.append("Kitchen light should be OFF")
        if hallway_light.is_on:
            errors.append("Hallway light should be OFF")
        
        # Baby monitoring
        if not nursery_camera.is_recording:
            errors.append("Nursery camera should still be recording")
        
        # Doorbell DND
        if not doorbell.do_not_disturb:
            errors.append("Doorbell should still be in DND mode")
        
        # Doors locked
        all_doors_locked = front_lock.is_locked and back_lock.is_locked
        if not front_lock.is_locked:
            errors.append("Front door should be locked")
        if not back_lock.is_locked:
            errors.append("Back door should be locked")
        
        if not errors:
            return ValidatorResult(True, f"Movie night complete!")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable("nursery_sensor", nursery_sensor,
             "Nursery temperature sensor"),
    Variable("nursery_camera", nursery_camera,
             "Nursery camera"),
    Variable("living_room_light", living_room_light,
             "Living room light"),
    Variable("living_room_blinds", living_room_blinds,
             "Living room blinds"),
    Variable("living_room_tv", living_room_tv,
             "Living room TV"),
    Variable("living_room_speaker", living_room_speaker,
             "Living room speaker"),
    Variable("kitchen_light", kitchen_light,
             "Kitchen light"),
    Variable("hallway_light", hallway_light,
             "Hallway light"),
    Variable("thermostat", thermostat,
             "Thermostat"),
    Variable("front_door_lock", front_door_lock,
             "Front door lock"),
    Variable("back_door_lock", back_door_lock,
             "Back door lock"),
    Variable("doorbell", doorbell,
             "Smart doorbell"),
]

types = [
    Type(Light),
    Type(TV),
    Type(Speaker),
    Type(Blinds),
    Type(Camera),
    Type(Lock),
    Type(Doorbell),
    Type(Thermostat),
    Type(TemperatureSensor),
]

validators = {
    "validate_baby_safety_setup": validate_baby_safety_setup,
    "validate_movie_setup": validate_movie_setup,
    "validate_movie_night_end": validate_movie_night_end,
}