"""
Work From Home Day - Multi-Turn Benchmark

Tests:
- Temperature range checking (below X OR above Y)
- Device coordination for video calls
- Tracking silenced devices in a list
- Conditional robot vacuum control
- Device counting/aggregation
"""

from typing import List
from cave_agent.python_runtime import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
from .types import Light, Thermostat, Lock, Camera, Speaker, Blinds, TemperatureSensor, SmartPlug, RobotVacuum, Doorbell


# ============================================================================
# INITIAL DEVICE STATE - Work from home morning
# ============================================================================

# Office devices
office_sensor = TemperatureSensor("Office Sensor", "Office", temperature=18.5, humidity=45)  # Cold!
office_light = Light("Office Light", "Office", is_on=False, brightness=0)
office_blinds = Blinds("Office Blinds", "Office", position=0)  # Closed overnight
office_plug = SmartPlug("Office Plug", "Office", is_on=False, connected_device="Monitor", power_draw=45)

# Living room devices
living_room_light = Light("Living Room Light", "Living Room", is_on=False, brightness=0)
living_room_blinds = Blinds("Living Room Blinds", "Living Room", position=0)
living_room_speaker = Speaker("Living Room Speaker", "Living Room", is_on=True)
living_room_speaker.is_playing = True  # Music was playing from morning routine
living_room_speaker.current_source = "morning playlist"
living_room_speaker.volume = 40

# Bedroom devices
bedroom_light = Light("Bedroom Light", "Bedroom", is_on=False, brightness=0)
bedroom_blinds = Blinds("Bedroom Blinds", "Bedroom", position=0)

# Climate
thermostat = Thermostat("Main Thermostat", "Hallway", current_temp=18, target_temp=18)

# Robot vacuum - cleaning from earlier morning routine!
robot_vacuum = RobotVacuum("Robot Vacuum", "Living Room", battery_level=75, dustbin_level=45)
robot_vacuum.is_cleaning = True
robot_vacuum.is_docked = False
robot_vacuum.is_on = True
robot_vacuum.current_room = "kitchen"

# Security
doorbell = Doorbell("Front Doorbell", "Front Door")
front_door_lock = Lock("Front Door", "Entrance", is_locked=True)
security_camera = Camera("Front Camera", "Front Door", is_on=True)


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_work_start(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 1: Work day setup.
    
    Initial state:
    - office_sensor.temperature = 18.5°C (below 20°C!)
    - office_light: off
    - office_blinds: position=0
    - office_plug: off
    - doorbell.do_not_disturb = False
    - robot_vacuum.is_cleaning = True (needs to stop!)
    
    Expected:
    - thermostat.target_temp = 22 (since 18.5 < 20)
    - office_light: on at 80%
    - office_blinds: position=70
    - office_plug: on
    - doorbell.do_not_disturb = True
    - robot_vacuum: docked and NOT cleaning
    """
    try:
        thermostat = runtime.retrieve("thermostat")
        office_light = runtime.retrieve("office_light")
        office_blinds = runtime.retrieve("office_blinds")
        office_plug = runtime.retrieve("office_plug")
        doorbell = runtime.retrieve("doorbell")
        vacuum = runtime.retrieve("robot_vacuum")
        
        errors = []
        
        # Temperature check: 18.5 < 20, so should set to 22
        if thermostat.target_temp != 22:
            errors.append(
                f"Thermostat should be 22°C (office was 18.5°C, below 20°C range), "
                f"got {thermostat.target_temp}°C"
            )
        
        # Office light at 80%
        if not office_light.is_on:
            errors.append("Office light should be ON")
        elif office_light.brightness != 80:
            errors.append(f"Office light should be 80%, got {office_light.brightness}%")
        
        # Office blinds at 70%
        if office_blinds.position != 70:
            errors.append(f"Office blinds should be at 70%, got {office_blinds.position}%")
        
        # Office plug on
        if not office_plug.is_on:
            errors.append("Office plug (monitor) should be ON")
        
        # Doorbell DND
        if not doorbell.do_not_disturb:
            errors.append("Doorbell should have do-not-disturb enabled")
        
        # Robot vacuum should be docked/not cleaning
        if vacuum.is_cleaning:
            errors.append("Robot vacuum should NOT be cleaning during work hours")
        if not vacuum.is_docked:
            errors.append("Robot vacuum should be docked")
        
        if not errors:
            return ValidatorResult(True, "Work setup complete! Office ready for productivity.")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


def validate_video_call_setup(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 2: Video call preparation.
    
    State after Turn 1:
    - vacuum should be docked
    - living_room_speaker was still playing (from initial state)
    
    Expected:
    - office_light: 100%
    - office_blinds: 50%
    - robot_vacuum: paused/not cleaning (was already docked from Turn 1)
    - living_room_speaker: stopped
    - doorbell: still in DND
    """
    try:
        office_light = runtime.retrieve("office_light")
        office_blinds = runtime.retrieve("office_blinds")
        vacuum = runtime.retrieve("robot_vacuum")
        speaker = runtime.retrieve("living_room_speaker")
        doorbell = runtime.retrieve("doorbell")
        
        errors = []
        
        # Office light at 100%
        if not office_light.is_on:
            errors.append("Office light should be ON")
        elif office_light.brightness != 100:
            errors.append(f"Office light should be 100% for video, got {office_light.brightness}%")
        
        # Office blinds at 50%
        if office_blinds.position != 50:
            errors.append(f"Office blinds should be at 50%, got {office_blinds.position}%")
        
        # Vacuum should not be cleaning
        if vacuum.is_cleaning:
            errors.append("Robot vacuum should not be cleaning")
        
        # Speaker should be stopped
        if speaker.is_playing:
            errors.append("Living room speaker should be stopped")
        
        # Doorbell still in DND
        if not doorbell.do_not_disturb:
            errors.append("Doorbell should still be in DND mode")
        
       
        if not errors:
            return ValidatorResult(True, f"Video call setup complete!")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


def validate_work_end(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 3: End of work day.
    
    Initial vacuum state: battery=75% (>60%), dustbin=45% (<70%)
    
    Expected:
    - office_plug: off
    - office_light: on at 30%
    - All blinds: open (position=100)
    - doorbell: DND disabled
    - robot_vacuum: cleaning living room in auto mode
    """
    try:
        office_plug = runtime.retrieve("office_plug")
        office_light = runtime.retrieve("office_light")
        office_blinds = runtime.retrieve("office_blinds")
        living_room_blinds = runtime.retrieve("living_room_blinds")
        bedroom_blinds = runtime.retrieve("bedroom_blinds")
        doorbell = runtime.retrieve("doorbell")
        vacuum = runtime.retrieve("robot_vacuum")
        
        errors = []
        
        # Office plug off
        if office_plug.is_on:
            errors.append("Office plug should be OFF")
        
        # Office light at 30%
        if not office_light.is_on:
            errors.append("Office light should be ON (dimmed for reading)")
        elif office_light.brightness != 30:
            errors.append(f"Office light should be 30%, got {office_light.brightness}%")
        
        # All blinds open
        if office_blinds.position != 100:
            errors.append(f"Office blinds should be fully open (100), got {office_blinds.position}")
        if living_room_blinds.position != 100:
            errors.append(f"Living room blinds should be fully open (100), got {living_room_blinds.position}")
        if bedroom_blinds.position != 100:
            errors.append(f"Bedroom blinds should be fully open (100), got {bedroom_blinds.position}")
        
        # Doorbell DND disabled
        if doorbell.do_not_disturb:
            errors.append("Doorbell DND should be disabled")
        
        # Vacuum should be cleaning (battery 75% > 60%, dustbin 45% < 70%)
        if not vacuum.is_cleaning:
            errors.append(
                "Robot vacuum should be cleaning (battery=75%>60%, dustbin=45%<70%)"
            )
        else:
            if vacuum.current_room is None or "living" not in vacuum.current_room.lower():
                errors.append(f"Vacuum should be cleaning living room, got {vacuum.current_room}")
            if vacuum.cleaning_mode != "auto":
                errors.append(f"Vacuum should be in auto mode, got {vacuum.cleaning_mode}")
        
       
        
        if not errors:
            return ValidatorResult(True, 
                f"Work day ended! Relaxation mode active.")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    Variable(
        "office_sensor", office_sensor,
        "Office temperature sensor. Props: temperature (float °C), humidity (int %). "
        "Access directly: office_sensor.temperature"
    ),
    Variable(
        "office_light", office_light,
        "Office light. Props: is_on, brightness (0-100). "
        "Methods: turn_on(), turn_off(), set_brightness(int)"
    ),
    Variable(
        "office_blinds", office_blinds,
        "Office blinds. Props: position (0=closed, 100=fully open). "
        "Methods: open(), close(), set_position(int)"
    ),
    Variable(
        "office_plug", office_plug,
        "Smart plug for office monitor. Props: is_on, connected_device, power_draw. "
        "Methods: turn_on(), turn_off()"
    ),
    Variable(
        "living_room_light", living_room_light,
        "Living room light. Props: is_on, brightness. Methods: turn_on(), turn_off(), set_brightness(int)"
    ),
    Variable(
        "living_room_blinds", living_room_blinds,
        "Living room blinds. Props: position. Methods: open(), close(), set_position(int)"
    ),
    Variable(
        "living_room_speaker", living_room_speaker,
        "Living room speaker. Props: is_on, is_playing, volume, current_source. "
        "Methods: turn_off(), play(source), stop(), set_volume(int)"
    ),
    Variable(
        "bedroom_light", bedroom_light,
        "Bedroom light. Props: is_on, brightness"
    ),
    Variable(
        "bedroom_blinds", bedroom_blinds,
        "Bedroom blinds. Props: position. Methods: open(), close(), set_position(int)"
    ),
    Variable(
        "thermostat", thermostat,
        "Thermostat. Props: current_temp, target_temp (int °C), mode. "
        "Methods: set_temperature(int), set_mode(str)"
    ),
    Variable(
        "robot_vacuum", robot_vacuum,
        "Robot vacuum. Props: battery_level (0-100%), dustbin_level (0-100%), is_cleaning, is_docked, current_room, cleaning_mode. "
        "Methods: start_cleaning(room: str, mode: str), pause(), resume(), return_to_dock(). "
        "Modes: 'auto', 'spot', 'edge', 'quiet'. Needs battery>20% and dustbin<100% to clean."
    ),
    Variable(
        "doorbell", doorbell,
        "Smart doorbell. Props: do_not_disturb (bool). "
        "Methods: enable_dnd(), disable_dnd()"
    ),
    Variable(
        "front_door_lock", front_door_lock,
        "Front door lock. Props: is_locked. Methods: lock(), unlock()"
    ),
    Variable(
        "security_camera", security_camera,
        "Security camera. Props: is_on, is_recording. Methods: start_recording(), stop_recording()"
    ),
]

types = [
    Type(Light),
    Type(Blinds),
    Type(TemperatureSensor),
    Type(Thermostat),
    Type(SmartPlug),
    Type(Speaker),
    Type(RobotVacuum),
    Type(Doorbell),
    Type(Lock),
    Type(Camera),
]

validators = {
    "validate_work_start": validate_work_start,
    "validate_video_call_setup": validate_video_call_setup,
    "validate_work_end": validate_work_end,
}

