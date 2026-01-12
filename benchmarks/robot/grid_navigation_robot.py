"""
Grid Navigation Robot - Benchmark Module

Tests CaveAgent's ability to:
- Navigate a 2D grid with obstacles
- Make movement decisions based on surroundings
- Collect items and manage inventory
- Calculate distances and optimize paths
- Manage battery/energy resources
- Execute conditional logic based on robot state
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import math

from cave_agent.python_runtime import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class Direction(Enum):
    """Cardinal directions for robot facing."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class CellType(Enum):
    """Types of cells in the grid."""
    EMPTY = "empty"
    OBSTACLE = "obstacle"
    CHARGER = "charger"
    BASE = "base"


@dataclass
class Item:
    """Collectible item in the grid."""
    name: str
    weight: float  # kg
    value: int  # points
    
    def __repr__(self):
        return f"Item({self.name}, {self.weight}kg, {self.value}pts)"


@dataclass
class Position:
    """2D position on the grid."""
    x: int
    y: int
    
    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        if isinstance(other, tuple) and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        return False
    
    def __repr__(self):
        return f"({self.x}, {self.y})"
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


# ============================================================================
# GRID WORLD CLASS
# ============================================================================

class GridWorld:
    """
    The environment grid where the robot operates.
    
    Coordinate system:
    - (0, 0) is bottom-left corner
    - x increases going right (east)
    - y increases going up (north)
    """
    
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.obstacles: Set[Tuple[int, int]] = set()
        self.items: Dict[Tuple[int, int], Item] = {}
        self.charger_position: Optional[Tuple[int, int]] = None
        self.base_position: Tuple[int, int] = (0, 0)
    
    def add_obstacle(self, x: int, y: int):
        """Add an obstacle at position."""
        self.obstacles.add((x, y))
    
    def add_item(self, x: int, y: int, item: Item):
        """Place an item at position."""
        self.items[(x, y)] = item
    
    def remove_item(self, x: int, y: int) -> Optional[Item]:
        """Remove and return item at position."""
        return self.items.pop((x, y), None)
    
    def set_charger(self, x: int, y: int):
        """Set charging station position."""
        self.charger_position = (x, y)
    
    def set_base(self, x: int, y: int):
        """Set base/home position."""
        self.base_position = (x, y)
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_obstacle(self, x: int, y: int) -> bool:
        """Check if position has an obstacle."""
        return (x, y) in self.obstacles
    
    def is_passable(self, x: int, y: int) -> bool:
        """Check if robot can move to position."""
        return self.is_valid_position(x, y) and not self.is_obstacle(x, y)
    
    def has_item(self, x: int, y: int) -> bool:
        """Check if position has an item."""
        return (x, y) in self.items
    
    def get_item(self, x: int, y: int) -> Optional[Item]:
        """Get item at position without removing it."""
        return self.items.get((x, y))
    
    def is_charger(self, x: int, y: int) -> bool:
        """Check if position is the charging station."""
        return self.charger_position == (x, y)
    
    def is_base(self, x: int, y: int) -> bool:
        """Check if position is the base."""
        return self.base_position == (x, y)
    
    def get_cell_type(self, x: int, y: int) -> str:
        """Get the type of cell at position."""
        if not self.is_valid_position(x, y):
            return "out_of_bounds"
        if self.is_obstacle(x, y):
            return "obstacle"
        if self.is_charger(x, y):
            return "charger"
        if self.is_base(x, y):
            return "base"
        return "empty"
    
    def get_all_item_positions(self) -> List[Tuple[int, int]]:
        """Get list of all positions containing items."""
        return list(self.items.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """Get grid world status."""
        return {
            "width": self.width,
            "height": self.height,
            "obstacle_count": len(self.obstacles),
            "item_count": len(self.items),
            "charger_position": self.charger_position,
            "base_position": self.base_position
        }
    
    def __repr__(self):
        return f"GridWorld({self.width}x{self.height}, {len(self.obstacles)} obstacles, {len(self.items)} items)"


# ============================================================================
# ROBOT CLASS
# ============================================================================

class Robot:
    """
    Grid navigation robot with movement, sensing, and collection capabilities.
    
    The robot can:
    - Move forward in its facing direction
    - Turn left or right (90 degrees)
    - Move directly to coordinates (if path is clear)
    - Scan surroundings
    - Pick up and carry items
    - Manage battery
    """
    
    def __init__(self, world: GridWorld, start_x: int = 0, start_y: int = 0, 
                 facing: str = "north", battery: int = 100):
        self.world = world
        self.x = start_x
        self.y = start_y
        self.facing = facing  # "north", "south", "east", "west"
        self.battery = battery  # 0-100 percentage
        self.inventory: List[Item] = []
        self.max_carry_weight = 10.0  # kg
        self.total_distance_moved = 0
        self.items_collected = 0
        self.is_charging = False
    
    # -------------------------
    # Position & State
    # -------------------------
    
    def get_position(self) -> Tuple[int, int]:
        """Get current position as (x, y) tuple."""
        return (self.x, self.y)
    
    def get_facing(self) -> str:
        """Get current facing direction."""
        return self.facing
    
    def get_battery(self) -> int:
        """Get current battery level (0-100)."""
        return self.battery
    
    def get_inventory(self) -> List[Item]:
        """Get list of items currently carried."""
        return self.inventory.copy()
    
    def get_inventory_weight(self) -> float:
        """Get total weight of carried items."""
        return sum(item.weight for item in self.inventory)
    
    def get_inventory_value(self) -> int:
        """Get total value of carried items."""
        return sum(item.value for item in self.inventory)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get complete robot status.
        Returns:
            dict: {
                "position": (x, y),
                "facing": "north"/"south"/"east"/"west",
                "battery": 0-100,
                "inventory_count": int,
                "inventory_weight": float,
                "inventory_value": int,
                "total_distance": int,
                "items_collected": int,
                "is_charging": bool
            }
        """
        return {
            "position": (self.x, self.y),
            "facing": self.facing,
            "battery": self.battery,
            "inventory_count": len(self.inventory),
            "inventory_weight": self.get_inventory_weight(),
            "inventory_value": self.get_inventory_value(),
            "total_distance": self.total_distance_moved,
            "items_collected": self.items_collected,
            "is_charging": self.is_charging
        }
    
    # -------------------------
    # Movement
    # -------------------------
    
    def _get_direction_delta(self, direction: str) -> Tuple[int, int]:
        """Get (dx, dy) for a direction."""
        deltas = {
            "north": (0, 1),
            "south": (0, -1),
            "east": (1, 0),
            "west": (-1, 0)
        }
        return deltas.get(direction, (0, 0))
    
    def move_forward(self) -> str:
        """
        Move one step in the facing direction.
        Returns status message.
        Consumes 1 battery per move.
        """
        if self.battery <= 0:
            return "Cannot move: battery depleted"
        
        dx, dy = self._get_direction_delta(self.facing)
        new_x, new_y = self.x + dx, self.y + dy
        
        if not self.world.is_valid_position(new_x, new_y):
            return f"Cannot move: out of bounds at ({new_x}, {new_y})"
        
        if self.world.is_obstacle(new_x, new_y):
            return f"Cannot move: obstacle at ({new_x}, {new_y})"
        
        self.x, self.y = new_x, new_y
        self.battery -= 1
        self.total_distance_moved += 1
        self.is_charging = False
        
        return f"Moved to ({self.x}, {self.y})"
    
    def move_steps(self, steps: int) -> str:
        """
        Move multiple steps in the facing direction.
        Stops if obstacle encountered.
        """
        moved = 0
        for _ in range(steps):
            result = self.move_forward()
            if "Cannot" in result:
                return f"Moved {moved} steps, then stopped: {result}"
            moved += 1
        return f"Moved {moved} steps to ({self.x}, {self.y})"
    
    def turn_left(self) -> str:
        """Turn 90 degrees left (counterclockwise)."""
        turns = {"north": "west", "west": "south", "south": "east", "east": "north"}
        self.facing = turns[self.facing]
        return f"Turned left, now facing {self.facing}"
    
    def turn_right(self) -> str:
        """Turn 90 degrees right (clockwise)."""
        turns = {"north": "east", "east": "south", "south": "west", "west": "north"}
        self.facing = turns[self.facing]
        return f"Turned right, now facing {self.facing}"
    
    def turn_to(self, direction: str) -> str:
        """Turn to face a specific direction."""
        if direction not in ["north", "south", "east", "west"]:
            return f"Invalid direction: {direction}"
        self.facing = direction
        return f"Now facing {self.facing}"
    
    def move_to(self, target_x: int, target_y: int) -> str:
        """
        Move directly to target position (straight-line movement).
        This is a simplified movement - moves in x then y.
        Consumes 1 battery per step.
        Returns result message.
        """
        if self.battery <= 0:
            return "Cannot move: battery depleted"
        
        if not self.world.is_valid_position(target_x, target_y):
            return f"Invalid target: ({target_x}, {target_y}) is out of bounds"
        
        if self.world.is_obstacle(target_x, target_y):
            return f"Invalid target: ({target_x}, {target_y}) is an obstacle"
        
        steps_taken = 0
        
        # Move in x direction first
        while self.x != target_x:
            if self.battery <= 0:
                return f"Battery depleted at ({self.x}, {self.y}) after {steps_taken} steps"
            
            dx = 1 if target_x > self.x else -1
            next_x = self.x + dx
            
            if self.world.is_obstacle(next_x, self.y):
                return f"Path blocked by obstacle at ({next_x}, {self.y}), stopped at ({self.x}, {self.y})"
            
            self.x = next_x
            self.battery -= 1
            self.total_distance_moved += 1
            steps_taken += 1
        
        # Move in y direction
        while self.y != target_y:
            if self.battery <= 0:
                return f"Battery depleted at ({self.x}, {self.y}) after {steps_taken} steps"
            
            dy = 1 if target_y > self.y else -1
            next_y = self.y + dy
            
            if self.world.is_obstacle(self.x, next_y):
                return f"Path blocked by obstacle at ({self.x}, {next_y}), stopped at ({self.x}, {self.y})"
            
            self.y = next_y
            self.battery -= 1
            self.total_distance_moved += 1
            steps_taken += 1
        
        self.is_charging = False
        return f"Arrived at ({self.x}, {self.y}) after {steps_taken} steps"
    
    # -------------------------
    # Sensing
    # -------------------------
    
    def scan_ahead(self) -> Dict[str, Any]:
        """
        Scan the cell directly ahead.
        Returns:
            dict: {
                "position": (x, y),
                "is_valid": bool,
                "is_passable": bool,
                "cell_type": str,
                "has_item": bool,
                "item": Item
            }
        """
        dx, dy = self._get_direction_delta(self.facing)
        ahead_x, ahead_y = self.x + dx, self.y + dy
        
        return {
            "position": (ahead_x, ahead_y),
            "is_valid": self.world.is_valid_position(ahead_x, ahead_y),
            "is_passable": self.world.is_passable(ahead_x, ahead_y),
            "cell_type": self.world.get_cell_type(ahead_x, ahead_y),
            "has_item": self.world.has_item(ahead_x, ahead_y),
            "item": self.world.get_item(ahead_x, ahead_y)
        }
    
    def scan_surroundings(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan all four adjacent cells.
        Returns:
            dict: {
                "position": (x, y),
                "is_passable": bool,
                "cell_type": str,
                "has_item": bool,
                "item": Item
            }
        """
        results = {}
        for direction in ["north", "south", "east", "west"]:
            dx, dy = self._get_direction_delta(direction)
            check_x, check_y = self.x + dx, self.y + dy
            results[direction] = {
                "position": (check_x, check_y),
                "is_passable": self.world.is_passable(check_x, check_y),
                "cell_type": self.world.get_cell_type(check_x, check_y),
                "has_item": self.world.has_item(check_x, check_y)
            }
        return results
    
    def scan_current(self) -> Dict[str, Any]:
        """Scan the current cell.
        Returns:
            dict: {
                "position": (x, y),
                "cell_type": str,
                "has_item": bool,
                "item": Item
            }
        """
        return {
            "position": (self.x, self.y),
            "cell_type": self.world.get_cell_type(self.x, self.y),
            "has_item": self.world.has_item(self.x, self.y),
            "item": self.world.get_item(self.x, self.y),
            "is_charger": self.world.is_charger(self.x, self.y),
            "is_base": self.world.is_base(self.x, self.y)
        }
    
    def detect_items_in_range(self, range_distance: int = 3) -> List[Dict[str, Any]]:
        """Detect all items within Manhattan distance range.
        Returns:
            list: [
                {
                    "position": (x, y),
                    "item": Item,
                    "distance": int
                }
            ]
        """
        items_found = []
        for (ix, iy), item in self.world.items.items():
            distance = abs(ix - self.x) + abs(iy - self.y)
            if distance <= range_distance:
                items_found.append({
                    "position": (ix, iy),
                    "item": item,
                    "distance": distance
                })
        return sorted(items_found, key=lambda x: x["distance"])
    
    # -------------------------
    # Item Interaction
    # -------------------------
    
    def pick_up(self) -> str:
        """Pick up item at current position.
        Returns:
            str: message indicating the item picked up
        """
        if not self.world.has_item(self.x, self.y):
            return "No item at current position"
        
        item = self.world.get_item(self.x, self.y)
        
        if self.get_inventory_weight() + item.weight > self.max_carry_weight:
            return f"Cannot pick up {item.name}: would exceed weight limit ({self.get_inventory_weight():.1f} + {item.weight:.1f} > {self.max_carry_weight})"
        
        # Remove from world and add to inventory
        self.world.remove_item(self.x, self.y)
        self.inventory.append(item)
        self.items_collected += 1
        
        return f"Picked up {item.name} (weight: {item.weight}kg, value: {item.value}pts)"
    
    def drop(self, item_name: str = None) -> str:
        """Drop an item at current position.
        Returns:
            str: message indicating the item dropped
        """
        if not self.inventory:
            return "Inventory is empty"
        
        if item_name:
            # Find specific item
            for i, item in enumerate(self.inventory):
                if item.name.lower() == item_name.lower():
                    dropped = self.inventory.pop(i)
                    self.world.add_item(self.x, self.y, dropped)
                    return f"Dropped {dropped.name}"
            return f"Item '{item_name}' not in inventory"
        else:
            # Drop first item
            dropped = self.inventory.pop(0)
            self.world.add_item(self.x, self.y, dropped)
            return f"Dropped {dropped.name}"
    
    def drop_all(self) -> str:
        """Drop all items at current position (for delivery).
        Returns:
            str: message indicating the items dropped
        """
        if not self.inventory:
            return "Inventory is empty"
        
        count = len(self.inventory)
        total_value = self.get_inventory_value()
        
        # In a real scenario, items would be delivered
        # For simplicity, we just clear inventory
        self.inventory.clear()
        
        return f"Delivered {count} items (total value: {total_value}pts)"
    
    # -------------------------
    # Battery Management
    # -------------------------
    
    def charge(self) -> str:
        """Charge battery if at charging station.
        Returns:
            str: message indicating the battery charged
        """
        if not self.world.is_charger(self.x, self.y):
            return "Not at charging station"
        
        self.battery = 100
        self.is_charging = True
        return "Battery fully charged (100%)"
    
    def partial_charge(self, amount: int) -> str:
        """Charge battery by a specific amount.
        Returns:
            str: message indicating the battery charged
        """
        if not self.world.is_charger(self.x, self.y):
            return "Not at charging station"
        
        old_battery = self.battery
        self.battery = min(100, self.battery + amount)
        self.is_charging = True
        return f"Battery charged from {old_battery}% to {self.battery}%"
    
    # -------------------------
    # Utility Methods
    # -------------------------
    
    def calculate_distance_to(self, target_x: int, target_y: int) -> int:
        """Calculate Manhattan distance to target.
        Returns:
            int: Manhattan distance to target
        """
        return abs(target_x - self.x) + abs(target_y - self.y)
    
    def calculate_euclidean_distance_to(self, target_x: int, target_y: int) -> float:
        """Calculate Euclidean (straight-line) distance to target.
        Returns:
            float: Euclidean distance to target
        """
        return math.sqrt((target_x - self.x) ** 2 + (target_y - self.y) ** 2)
    
    def can_reach(self, target_x: int, target_y: int) -> bool:
        """Check if robot has enough battery to reach target.
        Returns:
            bool: True if robot has enough battery to reach target
        """
        distance = self.calculate_distance_to(target_x, target_y)
        return self.battery >= distance
    
    def estimate_battery_for_trip(self, target_x: int, target_y: int) -> int:
        """Estimate battery needed for a trip to target and back to charger.
        Returns:
            int: estimated battery needed for a trip to target and back to charger
        """
        to_target = self.calculate_distance_to(target_x, target_y)
        if self.world.charger_position:
            cx, cy = self.world.charger_position
            target_to_charger = abs(target_x - cx) + abs(target_y - cy)
            return to_target + target_to_charger
        return to_target
    
    def __repr__(self):
        return f"Robot(pos=({self.x},{self.y}), facing={self.facing}, battery={self.battery}%, items={len(self.inventory)})"


# ============================================================================
# INITIAL WORLD SETUP
# ============================================================================

def create_test_world() -> Tuple[GridWorld, Robot]:
    """Create the test grid world and robot."""
    
    # Create 10x10 grid
    world = GridWorld(width=10, height=10)
    
    # Set base and charger
    world.set_base(0, 0)
    world.set_charger(9, 9)
    
    # Add obstacles (walls/barriers)
    # Vertical wall in middle with gap
    for y in range(3, 8):
        if y != 5:  # Gap at y=5
            world.add_obstacle(5, y)
    
    # Some scattered obstacles
    world.add_obstacle(2, 7)
    world.add_obstacle(7, 2)
    world.add_obstacle(8, 5)
    
    # Add items to collect
    world.add_item(3, 2, Item("Red Crystal", weight=1.0, value=10))
    world.add_item(7, 4, Item("Blue Gem", weight=2.0, value=25))
    world.add_item(1, 8, Item("Gold Coin", weight=0.5, value=15))
    world.add_item(6, 7, Item("Silver Key", weight=0.3, value=20))
    world.add_item(8, 1, Item("Green Orb", weight=1.5, value=30))
    
    # Create robot at base
    robot = Robot(world, start_x=0, start_y=0, facing="north", battery=50)
    
    return world, robot


# Create world and robot
world, robot = create_test_world()


# ============================================================================
# HELPER VARIABLES
# ============================================================================

# Track collected items value
total_collected_value = 0

# Track mission status
mission_complete = False

# List of items delivered to base
delivered_items: List[str] = []

mission_total_value = 0


# ============================================================================
# EXPORTS
# ============================================================================

tools = []



variables = [
    Variable(
        "robot", robot,
        "Navigation robot"
    ),
    Variable(
        "world", world,
        "10x10 grid world"
    ),
    Variable(
        "total_collected_value", total_collected_value,
        "Integer to track total value of items collected"
    ),
    Variable(
        "delivered_items", delivered_items,
        "List of item names that have been delivered to base"
    ),
    Variable(
        "mission_total_value", mission_total_value,
        "Integer to store total value of items collected during mission"
    ),
    Variable(
        "mission_complete", mission_complete,
        "Boolean flag - set to True when mission is finished"
    ),
]

types = [
    Type(Robot),
    Type(GridWorld),
    Type(Item),
]

def validate_first_collection(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 1: Find closest item and collect it.
    
    Initial state:
    - robot at (0,0), battery=50%
    - items in range(5): only Red Crystal at (3,2), distance=5
    
    Expected:
    - robot.position = (3, 2)
    - robot.inventory contains Red Crystal
    - robot.items_collected = 1
    - battery decreased (used ~5 for movement)
    """
    try:
        robot = runtime.retrieve("robot")
        world = runtime.retrieve("world")
        
        errors = []
        
        # Check robot moved to Red Crystal position (3, 2)
        pos = robot.get_position()
        if pos != (3, 2):
            errors.append(
                f"Robot should be at (3,2) - the closest item. "
                f"Got position {pos}"
            )
        
        # Check item was picked up
        if robot.items_collected < 1:
            errors.append("Robot should have collected at least 1 item")
        
        # Check inventory has the Red Crystal
        inventory = robot.get_inventory()
        has_red_crystal = any("red" in item.name.lower() or "crystal" in item.name.lower() 
                             for item in inventory)
        if not has_red_crystal:
            item_names = [item.name for item in inventory]
            errors.append(
                f"Robot should have picked up Red Crystal (closest item). "
                f"Inventory: {item_names}"
            )
        
        # Check item removed from world
        if world.has_item(3, 2):
            errors.append("Red Crystal should be removed from world at (3,2)")
        
        # Check battery was consumed (should be around 45%)
        battery = robot.get_battery()
        if battery >= 50:
            errors.append(f"Battery should have decreased from movement. Still at {battery}%")
        
        if not errors:
            return ValidatorResult(True, 
                f"First collection complete! Robot at {pos}, "
                f"collected {robot.items_collected} item(s), battery={battery}%")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


def validate_second_collection(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 2: Check battery, find highest value item, collect it.
    
    State after Turn 1:
    - robot at (3,2), battery ~45%
    - Red Crystal collected
    
    Remaining items (by value):
    - Green Orb (8,1): 30pts ← HIGHEST
    - Blue Gem (7,4): 25pts
    - Silver Key (6,7): 20pts  
    - Gold Coin (1,8): 15pts
    
    Expected:
    - Battery was >= 30, so NO charging needed
    - robot navigated to (8,1) to get Green Orb
    - robot.inventory now has Red Crystal + Green Orb
    - robot.items_collected = 2
    """
    try:
        robot = runtime.retrieve("robot")
        world = runtime.retrieve("world")
        
        errors = []
        
        # Check robot went for the highest value item (Green Orb at 8,1)
        pos = robot.get_position()
        if pos != (8, 1):
            errors.append(
                f"Robot should be at (8,1) - location of highest value item (Green Orb, 30pts). "
                f"Got position {pos}"
            )
        
        # Check Green Orb was picked up
        inventory = robot.get_inventory()
        has_green_orb = any("green" in item.name.lower() or "orb" in item.name.lower() 
                           for item in inventory)
        if not has_green_orb:
            item_names = [item.name for item in inventory]
            errors.append(
                f"Robot should have picked up Green Orb (highest value). "
                f"Inventory: {item_names}"
            )
        
        # Check total items collected
        if robot.items_collected < 2:
            errors.append(f"Robot should have collected 2 items by now. Got {robot.items_collected}")
        
        # Check inventory has both items (Red Crystal from Turn 1 + Green Orb)
        if len(inventory) < 2:
            errors.append(f"Inventory should have 2 items. Got {len(inventory)}")
        
        # Check Green Orb removed from world
        if world.has_item(8, 1):
            errors.append("Green Orb should be removed from world at (8,1)")
        
        # Verify inventory value (should be 10 + 30 = 40)
        total_value = robot.get_inventory_value()
        if total_value != 40:
            errors.append(
                f"Inventory value should be 40 (Red Crystal 10 + Green Orb 30). "
                f"Got {total_value}"
            )
        
        # Verify inventory weight (should be 1.0 + 1.5 = 2.5)
        total_weight = robot.get_inventory_weight()
        if abs(total_weight - 2.5) > 0.1:
            errors.append(
                f"Inventory weight should be ~2.5kg. Got {total_weight}kg"
            )
        
        if not errors:
            return ValidatorResult(True, 
                f"Second collection complete! At {pos}, "
                f"inventory: {total_value}pts, {total_weight}kg")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


def validate_mission_complete(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 3: Return to base and deliver items.
    
    State after Turn 2:
    - robot at (8,1), battery ~39%
    - inventory: Red Crystal + Green Orb (40pts total)
    
    Distance to base: |8-0| + |1-0| = 9 steps
    Battery 39 >= 9, so CAN reach base directly
    
    Expected:
    - robot.position = (0, 0) - back at base
    - robot.inventory = [] (delivered via drop_all)
    - mission_total_value = 40
    - mission_complete = True
    """
    try:
        robot = runtime.retrieve("robot")
        mission_value = runtime.retrieve("mission_total_value")
        mission_done = runtime.retrieve("mission_complete")
        
        errors = []
        
        # Check robot returned to base
        pos = robot.get_position()
        if pos != (0, 0):
            errors.append(
                f"Robot should be at base (0,0). Got position {pos}"
            )
        
        # Check inventory was delivered (should be empty)
        inventory = robot.get_inventory()
        if len(inventory) > 0:
            item_names = [item.name for item in inventory]
            errors.append(
                f"Robot should have delivered all items (drop_all). "
                f"Still has: {item_names}"
            )
        
        # Check mission_total_value
        if mission_value != 40:
            errors.append(
                f"mission_total_value should be 40 (Red Crystal 10 + Green Orb 30). "
                f"Got {mission_value}"
            )
        
        # Check mission_complete flag
        if mission_done != True:
            errors.append(
                f"mission_complete should be True. Got {mission_done}"
            )
        
        # Check robot has collected 2 items total during mission
        if robot.items_collected != 2:
            errors.append(
                f"Robot should have collected exactly 2 items. Got {robot.items_collected}"
            )
        
        if not errors:
            return ValidatorResult(True, 
                f"Mission complete! Delivered {mission_value}pts worth of items to base.")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


# Add to exports
validators = {
    "validate_first_collection": validate_first_collection,
    "validate_second_collection": validate_second_collection,
    "validate_mission_complete": validate_mission_complete,
}