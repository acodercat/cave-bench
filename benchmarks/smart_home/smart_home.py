"""
Smart Home Benchmark - Object Manipulation and Automation Logic

This module provides smart home device classes for testing CaveAgent's ability to:
- Manipulate objects and their properties
- Call methods on objects
- Implement automation logic
- Work with class hierarchies and inheritance
"""

from typing import Dict, List, Any
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# SMART DEVICE CLASSES
# ============================================================================

class SmartDevice:
    """Base class for all smart home devices."""

    def __init__(self, name: str, location: str, is_on: bool = False):
        self.name = name
        self.location = location
        self.is_on = is_on
        self.power_watts = 0  # Power consumption when on

    def turn_on(self):
        """Turn the device on."""
        self.is_on = True

    def turn_off(self):
        """Turn the device off."""
        self.is_on = False

    def toggle(self):
        """Toggle device on/off state."""
        self.is_on = not self.is_on

    def get_status(self) -> Dict[str, Any]:
        """Get current device status."""
        return {
            "name": self.name,
            "location": self.location,
            "is_on": self.is_on,
            "power_watts": self.power_watts if self.is_on else 0
        }

    def __repr__(self):
        status = "ON" if self.is_on else "OFF"
        return f"{self.name} ({self.location}): {status}"


class Light(SmartDevice):
    """Smart light with brightness and color temperature control."""

    def __init__(self, name: str, location: str, is_on: bool = False,
                 brightness: int = 100, color_temp: int = 4000):
        super().__init__(name, location, is_on)
        self.brightness = brightness  # 0-100
        self.color_temp = color_temp  # Kelvin (2700-6500)
        self.max_power_watts = 10  # LED bulb max power

    def set_brightness(self, brightness: int):
        """Set brightness level (0-100)."""
        self.brightness = max(0, min(100, brightness))
        if self.brightness > 0:
            self.is_on = True
        else:
            self.is_on = False
        self._update_power()

    def set_color_temp(self, kelvin: int):
        """Set color temperature in Kelvin (2700-6500)."""
        self.color_temp = max(2700, min(6500, kelvin))

    def dim(self, amount: int = 20):
        """Dim the light by specified amount."""
        self.set_brightness(self.brightness - amount)

    def brighten(self, amount: int = 20):
        """Brighten the light by specified amount."""
        self.set_brightness(self.brightness + amount)

    def _update_power(self):
        """Update power consumption based on brightness."""
        if self.is_on:
            self.power_watts = (self.brightness / 100) * self.max_power_watts
        else:
            self.power_watts = 0

    def turn_on(self):
        """Turn the light on at current brightness."""
        self.is_on = True
        if self.brightness == 0:
            self.brightness = 100
        self._update_power()

    def turn_off(self):
        """Turn the light off."""
        self.is_on = False
        self._update_power()

    def get_status(self) -> Dict[str, Any]:
        """Get current light status."""
        return {
            **super().get_status(),
            "brightness": self.brightness,
            "color_temp": self.color_temp
        }

    def __repr__(self):
        if self.is_on:
            return f"{self.name} ({self.location}): ON @ {self.brightness}% brightness"
        else:
            return f"{self.name} ({self.location}): OFF"


class Thermostat(SmartDevice):
    """Smart thermostat with temperature and mode control."""

    def __init__(self, name: str, location: str, current_temp: int = 72,
                 target_temp: int = 72, mode: str = "auto"):
        super().__init__(name, location, is_on=True)
        self.current_temp = current_temp  # Current temperature in Fahrenheit
        self.target_temp = target_temp    # Target temperature
        self.mode = mode  # "heat", "cool", "auto", "eco", "off"
        self.is_heating = False
        self.is_cooling = False

    def set_temperature(self, temp: int):
        """Set target temperature."""
        self.target_temp = temp
        self._update_hvac_state()

    def set_mode(self, mode: str):
        """Set thermostat mode: heat, cool, auto, eco, off."""
        self.mode = mode
        if mode == "off":
            self.is_on = False
            self.is_heating = False
            self.is_cooling = False
        else:
            self.is_on = True
            self._update_hvac_state()

    def _update_hvac_state(self):
        """Update heating/cooling state based on current conditions."""
        if not self.is_on or self.mode == "off":
            self.is_heating = False
            self.is_cooling = False
            self.power_watts = 0
            return

        temp_diff = self.target_temp - self.current_temp

        # Eco mode has wider tolerance (±3 degrees)
        tolerance = 3 if self.mode == "eco" else 1

        if self.mode == "heat":
            self.is_heating = temp_diff > tolerance
            self.is_cooling = False
        elif self.mode == "cool":
            self.is_heating = False
            self.is_cooling = temp_diff < -tolerance
        elif self.mode in ["auto", "eco"]:
            if temp_diff > tolerance:
                self.is_heating = True
                self.is_cooling = False
            elif temp_diff < -tolerance:
                self.is_heating = False
                self.is_cooling = True
            else:
                self.is_heating = False
                self.is_cooling = False

        # Update power consumption
        if self.is_heating or self.is_cooling:
            self.power_watts = 3000  # 3 kW for HVAC
        else:
            self.power_watts = 5  # Idle power

    def simulate_temperature_change(self, minutes: int):
        """Simulate temperature change over time."""
        # Heating/cooling: 1°F per 10 minutes
        # Natural drift: towards outside temp
        if self.is_heating:
            degrees = minutes / 10
            self.current_temp = min(self.target_temp, self.current_temp + degrees)
        elif self.is_cooling:
            degrees = minutes / 10
            self.current_temp = max(self.target_temp, self.current_temp - degrees)

        self._update_hvac_state()

    def get_status(self) -> Dict[str, Any]:
        """Get current thermostat status."""
        return {
            **super().get_status(),
            "current_temp": self.current_temp,
            "target_temp": self.target_temp,
            "mode": self.mode,
            "is_heating": self.is_heating,
            "is_cooling": self.is_cooling
        }

    def __repr__(self):
        action = "HEATING" if self.is_heating else "COOLING" if self.is_cooling else "IDLE"
        return f"{self.name} ({self.location}): {self.current_temp}°F → {self.target_temp}°F [{action}]"


class Lock(SmartDevice):
    """Smart door lock."""

    def __init__(self, name: str, location: str, is_locked: bool = True):
        super().__init__(name, location, is_on=True)
        self.is_locked = is_locked
        self.auto_lock = True
        self.power_watts = 2  # Low power consumption

    def lock(self):
        """Lock the door."""
        self.is_locked = True

    def unlock(self):
        """Unlock the door."""
        self.is_locked = False

    def toggle_lock(self):
        """Toggle lock state."""
        self.is_locked = not self.is_locked

    def get_status(self) -> Dict[str, Any]:
        """Get current lock status."""
        return {
            **super().get_status(),
            "is_locked": self.is_locked,
            "auto_lock": self.auto_lock
        }

    def __repr__(self):
        status = "LOCKED" if self.is_locked else "UNLOCKED"
        return f"{self.name} ({self.location}): {status}"


class Camera(SmartDevice):
    """Security camera with motion detection."""

    def __init__(self, name: str, location: str, is_on: bool = True):
        super().__init__(name, location, is_on)
        self.is_recording = False
        self.motion_detected = False
        self.night_vision = True
        self.power_watts = 5

    def start_recording(self):
        """Start recording."""
        if self.is_on:
            self.is_recording = True

    def stop_recording(self):
        """Stop recording."""
        self.is_recording = False

    def detect_motion(self, detected: bool):
        """Set motion detection state."""
        self.motion_detected = detected
        if detected and self.is_on:
            self.is_recording = True  # Auto-record on motion

    def get_status(self) -> Dict[str, Any]:
        """Get current camera status."""
        return {
            **super().get_status(),
            "is_recording": self.is_recording,
            "motion_detected": self.motion_detected,
            "night_vision": self.night_vision
        }

    def __repr__(self):
        recording = "RECORDING" if self.is_recording else "STANDBY"
        motion = " [MOTION]" if self.motion_detected else ""
        return f"{self.name} ({self.location}): {recording}{motion}"


class Speaker(SmartDevice):
    """Smart speaker with volume control."""

    def __init__(self, name: str, location: str, is_on: bool = False):
        super().__init__(name, location, is_on)
        self.volume = 50  # 0-100
        self.is_playing = False
        self.current_source = None
        self.max_power_watts = 20

    def set_volume(self, volume: int):
        """Set volume level (0-100)."""
        self.volume = max(0, min(100, volume))
        self._update_power()

    def play(self, source: str = "music"):
        """Start playing."""
        self.is_on = True
        self.is_playing = True
        self.current_source = source
        self._update_power()

    def pause(self):
        """Pause playback."""
        self.is_playing = False
        self._update_power()

    def stop(self):
        """Stop playback."""
        self.is_playing = False
        self.current_source = None
        self._update_power()

    def _update_power(self):
        """Update power consumption."""
        if self.is_on and self.is_playing:
            self.power_watts = (self.volume / 100) * self.max_power_watts
        else:
            self.power_watts = 2 if self.is_on else 0

    def get_status(self) -> Dict[str, Any]:
        """Get current speaker status."""
        return {
            **super().get_status(),
            "volume": self.volume,
            "is_playing": self.is_playing,
            "current_source": self.current_source
        }

    def __repr__(self):
        if self.is_playing:
            return f"{self.name} ({self.location}): PLAYING {self.current_source} @ {self.volume}%"
        else:
            return f"{self.name} ({self.location}): OFF"


# ============================================================================
# SCENE AND AUTOMATION CLASSES
# ============================================================================

class Room:
    """Represents a room with devices."""

    def __init__(self, name: str):
        self.name = name
        self.devices = []

    def add_device(self, device: SmartDevice):
        """Add a device to the room."""
        self.devices.append(device)

    def turn_all_off(self):
        """Turn off all devices in the room."""
        for device in self.devices:
            device.turn_off()

    def turn_all_on(self):
        """Turn on all devices in the room."""
        for device in self.devices:
            device.turn_on()

    def get_total_power(self) -> float:
        """Get total power consumption of all devices."""
        return sum(d.power_watts for d in self.devices if d.is_on)

    def __repr__(self):
        return f"Room({self.name}): {len(self.devices)} devices"


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
thermostat = Thermostat("Main Thermostat", "Hallway", current_temp=72, target_temp=72)

# Security devices
front_door_lock = Lock("Front Door Lock", "Front Door", is_locked=True)
back_door_lock = Lock("Back Door Lock", "Back Door", is_locked=False)
security_camera = Camera("Front Camera", "Front Door", is_on=True)

# All devices list
all_devices = [
    living_room_light, living_room_speaker,
    bedroom_light, bedroom_speaker,
    kitchen_light, hallway_light,
    thermostat, front_door_lock, back_door_lock,
    security_camera
]

# Rooms
living_room = Room("Living Room")
living_room.add_device(living_room_light)
living_room.add_device(living_room_speaker)

bedroom = Room("Bedroom")
bedroom.add_device(bedroom_light)
bedroom.add_device(bedroom_speaker)

kitchen = Room("Kitchen")
kitchen.add_device(kitchen_light)


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
        lr_light = runtime.get_variable_value("living_room_light")

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
    """Validate leaving home automation - doors locked, lights off, camera recording, power calculated."""
    try:
        front_lock = runtime.get_variable_value("front_door_lock")
        back_lock = runtime.get_variable_value("back_door_lock")
        all_devices = runtime.get_variable_value("all_devices")
        security_camera = runtime.get_variable_value("security_camera")
        total_power = runtime.get_variable_value("total_power_watts")

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

        # Check 4: Power consumption calculated
        if total_power is None:
            errors.append("Total power not calculated")
        elif total_power <= 0:
            errors.append("Total power should be > 0 (camera + thermostat + locks are on)")

        if not errors:
            return ValidatorResult(True, f"Leaving home automation complete! Total power: {total_power}W")
        else:
            return ValidatorResult(False, f"Issues found: {'; '.join(errors)}")

    except KeyError as e:
        return ValidatorResult(False, f"Variable not found: {str(e)}")
    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


# ============================================================================
# EXPORTS
# ============================================================================

tools = []  # No function tools - object manipulation only
variables = [
    Variable("living_room_light", living_room_light, "Smart light in living room", include_type_doc=True, include_type_schema=True),
    Variable("bedroom_light", bedroom_light, "Smart light in bedroom", include_type_doc=True, include_type_schema=True),
    Variable("kitchen_light", kitchen_light, "Smart light in kitchen", include_type_doc=True, include_type_schema=True),
    Variable("hallway_light", hallway_light, "Smart light in hallway", include_type_doc=True, include_type_schema=True),
    Variable("living_room_speaker", living_room_speaker, "Smart speaker in living room", include_type_doc=True, include_type_schema=True),
    Variable("bedroom_speaker", bedroom_speaker, "Smart speaker in bedroom", include_type_doc=True, include_type_schema=True),
    Variable("thermostat", thermostat, "Main thermostat for home", include_type_doc=True, include_type_schema=True),
    Variable("front_door_lock", front_door_lock, "Front door smart lock", include_type_doc=True, include_type_schema=True),
    Variable("back_door_lock", back_door_lock, "Back door smart lock", include_type_doc=True, include_type_schema=True),
    Variable("security_camera", security_camera, "Security camera at front door", include_type_doc=True, include_type_schema=True),
    Variable("all_devices", all_devices, "List of all smart devices", include_type_doc=True, include_type_schema=True),
    Variable("living_room", living_room, "Living room with devices", include_type_doc=True, include_type_schema=True),
    Variable("bedroom", bedroom, "Bedroom with devices", include_type_doc=True, include_type_schema=True),
    Variable("kitchen", kitchen, "Kitchen with devices", include_type_doc=True, include_type_schema=True),
]

validators = {
    "validate_movie_mode": validate_movie_mode,
    "validate_leaving_home": validate_leaving_home,
}
