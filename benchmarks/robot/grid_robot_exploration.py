"""
Grid Robot Exploration - Step-by-Step Discovery Benchmark

Tests CaveAgent's ability to:
- Implement exploration loops with incremental movement
- Build and maintain a discovery map (data structure construction)
- Make navigation decisions based on partial information
- Use discovered information for pathfinding
- Manage resources (battery) during exploration
- Handle RANDOMNESS in movement outcomes

Key Innovations:
1. Robot has NO initial knowledge of the world - must discover through exploration
2. Movement has RANDOMNESS:
   - 20% chance of slipping (move perpendicular to intended direction)
   - Battery cost varies: 1-2 per step
   - Agent must check results and handle unexpected outcomes
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import random
from cave_agent.python_runtime import Variable, PythonRuntime, Type
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Item:
    """Collectible item in the grid."""
    name: str
    weight: float  # kg
    value: int  # points
    
    def __repr__(self):
        return f"Item({self.name}, {self.weight}kg, {self.value}pts)"


@dataclass 
class CellInfo:
    """Information about a discovered cell."""
    cell_type: str  # "empty", "obstacle", "charger", "base"
    has_item: bool
    item: Optional[Item]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_type": self.cell_type,
            "has_item": self.has_item,
            "item_name": self.item.name if self.item else None
        }
    
    def __repr__(self):
        if self.has_item:
            return f"Cell({self.cell_type}, item={self.item.name})"
        return f"Cell({self.cell_type})"


# ============================================================================
# HIDDEN WORLD (Ground Truth - Robot Cannot Access Directly)
# ============================================================================

class GridWorld:
    """
    The actual world state - robot cannot access this directly.
    Robot must discover this information through exploration.
    
    Coordinate system:
    - (0, 0) is bottom-left corner
    - x increases going right (east)
    - y increases going up (north)
    """
    
    def __init__(self, width: int = 8, height: int = 8):
        self.width = width
        self.height = height
        self._obstacles: Set[Tuple[int, int]] = set()
        self._items: Dict[Tuple[int, int], Item] = {}
        self._charger_position: Optional[Tuple[int, int]] = None
        self._base_position: Tuple[int, int] = (0, 0)
    
    def _add_obstacle(self, x: int, y: int):
        """Internal: Add obstacle."""
        self._obstacles.add((x, y))
    
    def _add_item(self, x: int, y: int, item: Item):
        """Internal: Add item."""
        self._items[(x, y)] = item
    
    def _remove_item(self, x: int, y: int) -> Optional[Item]:
        """Internal: Remove and return item."""
        return self._items.pop((x, y), None)
    
    def _set_charger(self, x: int, y: int):
        """Internal: Set charger position."""
        self._charger_position = (x, y)
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def _is_obstacle(self, x: int, y: int) -> bool:
        """Check if position has obstacle."""
        return (x, y) in self._obstacles
    
    def _is_passable(self, x: int, y: int) -> bool:
        """Check if robot can move to position."""
        return self._is_valid_position(x, y) and not self._is_obstacle(x, y)
    
    def _get_cell_info(self, x: int, y: int) -> Optional[CellInfo]:
        """Get full cell information (for robot discovery)."""
        if not self._is_valid_position(x, y):
            return None
        
        if self._is_obstacle(x, y):
            return CellInfo("obstacle", False, None)
        
        cell_type = "empty"
        if (x, y) == self._base_position:
            cell_type = "base"
        elif (x, y) == self._charger_position:
            cell_type = "charger"
        
        item = self._items.get((x, y))
        return CellInfo(cell_type, item is not None, item)
    
    def __repr__(self):
        return f"HiddenWorld({self.width}x{self.height})"


# ============================================================================
# EXPLORATION ROBOT
# ============================================================================

class Robot:
    """
    Exploration robot that discovers the world step-by-step.

    Key concept: Robot has LIMITED VISION
    - Can only see current cell and adjacent cells (4 directions)
    - Must build map through exploration
    - Cannot access world information directly

    Movement: ONE step per turn using step(direction)
    Sensing: scan_current(), scan_adjacent()

    IMPORTANT: Robot can only move ONCE per turn!
    After moving, you must wait for next turn to move again.
    """

    def __init__(self, world: GridWorld, start_x: int = 0, start_y: int = 0,
                 battery: int = 100):
        self._world = world  # Hidden - robot uses sensing methods
        self.x = start_x
        self.y = start_y
        self.facing = "north"
        self.battery = battery
        self.inventory: List[Item] = []
        self.max_carry_weight = 10.0

        # Statistics
        self.steps_taken = 0
        self.items_collected = 0

        # Turn-based movement limit
        self.moves_this_turn = 0
        self.max_moves_per_turn = 1  # Only ONE move per turn!

        # Last move result (for hooks to report)
        self.last_move_result: Optional[Dict[str, Any]] = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # POSITION & STATUS
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_position(self) -> Tuple[int, int]:
        """Get current position as (x, y) tuple."""
        return (self.x, self.y)
    
    def get_battery(self) -> int:
        """Get current battery level (0-100)."""
        return self.battery
    
    def get_inventory(self) -> List[Item]:
        """Get list of carried items."""
        return self.inventory.copy()
    
    def get_inventory_weight(self) -> float:
        """Get total weight of carried items."""
        return sum(item.weight for item in self.inventory)
    
    def get_inventory_value(self) -> int:
        """Get total value of carried items."""
        return sum(item.value for item in self.inventory)
    
    def get_status(self) -> Dict[str, Any]:
        """Get robot status summary.
        Returns:
            dict: {
                "position": (x, y),
                "facing": "north"/"south"/"east"/"west",
                "battery": 0-100,
                "inventory_count": int,
                "inventory_weight": float,
                "inventory_value": int,
            }
        """
        return {
            "position": (self.x, self.y),
            "facing": self.facing,
            "battery": self.battery,
            "inventory_count": len(self.inventory),
            "inventory_weight": self.get_inventory_weight(),
            "inventory_value": self.get_inventory_value(),
            "steps_taken": self.steps_taken,
            "items_collected": self.items_collected
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # SENSING - How robot discovers the world
    # ═══════════════════════════════════════════════════════════════════════
    
    def scan_current(self) -> Dict[str, Any]:
        """
        Scan the current cell.
        Returns information about what's at robot's position.
        Returns:
            dict: {
                "position": (x, y),
                "cell_type": "empty"/"obstacle"/"charger"/"base",
                "has_item": bool,
                "item": Item
            }
        """
        cell_info = self._world._get_cell_info(self.x, self.y)
        return {
            "position": (self.x, self.y),
            "cell_type": cell_info.cell_type,
            "has_item": cell_info.has_item,
            "item": cell_info.item
        }
    
    def scan_adjacent(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan all 4 adjacent cells (north, south, east, west).
        Returns what robot can see in each direction.
        This is robot's primary way to discover the world.
        Returns:
            dict: {
                "north": {
                    "position": (x, y),
                    "cell_type": "empty"/"obstacle"/"charger"/"base",
                    "is_passable": bool,
                    "has_item": bool,
                    "item": Item
                }
                "south": {
                    "position": (x, y),
                    "cell_type": "empty"/"obstacle"/"charger"/"base",
                    "is_passable": bool,
                    "has_item": bool,
                    "item": Item
                }
                "east": {
                    "position": (x, y),
                    "cell_type": "empty"/"obstacle"/"charger"/"base",
                    "is_passable": bool,
                    "has_item": bool,
                    "item": Item
                }
                "west": {
                    "position": (x, y),
                    "cell_type": "empty"/"obstacle"/"charger"/"base",
                    "is_passable": bool,
                    "has_item": bool,
                    "item": Item
                }
            }
        """
        directions = {
            "north": (0, 1),
            "south": (0, -1),
            "east": (1, 0),
            "west": (-1, 0)
        }
        
        results = {}
        for direction, (dx, dy) in directions.items():
            check_x, check_y = self.x + dx, self.y + dy
            
            if not self._world._is_valid_position(check_x, check_y):
                results[direction] = {
                    "position": (check_x, check_y),
                    "cell_type": "out_of_bounds",
                    "is_passable": False,
                    "has_item": False,
                    "item": None
                }
            else:
                cell_info = self._world._get_cell_info(check_x, check_y)
                results[direction] = {
                    "position": (check_x, check_y),
                    "cell_type": cell_info.cell_type,
                    "is_passable": cell_info.cell_type != "obstacle",
                    "has_item": cell_info.has_item,
                    "item": cell_info.item
                }
        
        return results
    
    # ═══════════════════════════════════════════════════════════════════════
    # MOVEMENT - Step by step exploration
    # ═══════════════════════════════════════════════════════════════════════
    
    def step(self, direction: str) -> Dict[str, Any]:
        """
        Take ONE step in the specified direction.
        This is the primary movement method for exploration.

        IMPORTANT: You can only move ONCE per turn!
        After calling step(), you must wait for the next turn to move again.

        WARNING: Movement has randomness!
        - 20% chance of slipping (end up in perpendicular direction)
        - Battery cost varies: 1-2 per step
        - Always check the result to see where you actually ended up!

        Args:
            direction: "north", "south", "east", or "west"

        Returns:
            Dict with movement result:
            - success: bool
            - intended_direction: str (what you requested)
            - actual_direction: str (where you actually moved, may differ if slipped)
            - slipped: bool (True if slip occurred)
            - position: (x, y) tuple of final position
            - battery_cost: int (1 or 2)
            - cell_type, has_item, item: info about destination cell
        """
        # Check turn limit
        if self.moves_this_turn >= self.max_moves_per_turn:
            result = {
                "success": False,
                "reason": "turn_limit_reached",
                "message": "Cannot move: already moved this turn. Wait for next turn.",
                "position": (self.x, self.y),
                "intended_direction": direction,
                "actual_direction": None,
                "slipped": False,
                "battery_cost": 0
            }
            self.last_move_result = result
            return result

        if self.battery <= 0:
            result = {
                "success": False,
                "reason": "battery_depleted",
                "message": "Cannot move: battery is empty",
                "position": (self.x, self.y),
                "intended_direction": direction,
                "actual_direction": None,
                "slipped": False,
                "battery_cost": 0
            }
            self.last_move_result = result
            return result

        direction_deltas = {
            "north": (0, 1),
            "south": (0, -1),
            "east": (1, 0),
            "west": (-1, 0)
        }

        # Perpendicular directions for slip mechanic
        perpendicular = {
            "north": ["east", "west"],
            "south": ["east", "west"],
            "east": ["north", "south"],
            "west": ["north", "south"]
        }

        if direction not in direction_deltas:
            result = {
                "success": False,
                "reason": "invalid_direction",
                "message": f"Invalid direction: {direction}. Use north/south/east/west.",
                "position": (self.x, self.y),
                "intended_direction": direction,
                "actual_direction": None,
                "slipped": False,
                "battery_cost": 0
            }
            self.last_move_result = result
            return result

        # Count this as a move attempt
        self.moves_this_turn += 1

        # Determine actual direction (20% slip chance)
        slipped = False
        actual_direction = direction
        if random.random() < 0.2:
            slipped = True
            actual_direction = random.choice(perpendicular[direction])

        dx, dy = direction_deltas[actual_direction]
        new_x, new_y = self.x + dx, self.y + dy

        # Check bounds
        if not self._world._is_valid_position(new_x, new_y):
            # If slipped into wall, stay in place but still lose battery
            battery_cost = random.randint(1, 2)
            self.battery -= battery_cost
            result = {
                "success": False,
                "reason": "out_of_bounds",
                "message": f"Tried to move {direction}" + (f" but slipped {actual_direction}" if slipped else "") + f": hit boundary at ({new_x}, {new_y})",
                "position": (self.x, self.y),
                "intended_direction": direction,
                "actual_direction": actual_direction,
                "slipped": slipped,
                "battery_cost": battery_cost
            }
            self.last_move_result = result
            return result

        # Check obstacle
        if self._world._is_obstacle(new_x, new_y):
            # If slipped into obstacle, stay in place but still lose battery
            battery_cost = random.randint(1, 2)
            self.battery -= battery_cost
            result = {
                "success": False,
                "reason": "obstacle",
                "message": f"Tried to move {direction}" + (f" but slipped {actual_direction}" if slipped else "") + f": obstacle at ({new_x}, {new_y})",
                "position": (self.x, self.y),
                "intended_direction": direction,
                "actual_direction": actual_direction,
                "slipped": slipped,
                "battery_cost": battery_cost
            }
            self.last_move_result = result
            return result

        # Move successful
        self.x, self.y = new_x, new_y
        battery_cost = random.randint(1, 2)
        self.battery -= battery_cost
        self.steps_taken += 1
        self.facing = actual_direction

        # Get info about new cell
        cell_info = self._world._get_cell_info(self.x, self.y)

        slip_msg = f" (slipped from {direction}!)" if slipped else ""
        result = {
            "success": True,
            "reason": "moved",
            "message": f"Moved {actual_direction} to ({self.x}, {self.y}){slip_msg}. Battery cost: {battery_cost}",
            "position": (self.x, self.y),
            "intended_direction": direction,
            "actual_direction": actual_direction,
            "slipped": slipped,
            "battery_cost": battery_cost,
            "cell_type": cell_info.cell_type,
            "has_item": cell_info.has_item,
            "item": cell_info.item
        }
        self.last_move_result = result
        return result
    
    def step_north(self) -> Dict[str, Any]:
        """Take one step north (y+1)."""
        return self.step("north")
    
    def step_south(self) -> Dict[str, Any]:
        """Take one step south (y-1)."""
        return self.step("south")
    
    def step_east(self) -> Dict[str, Any]:
        """Take one step east (x+1)."""
        return self.step("east")
    
    def step_west(self) -> Dict[str, Any]:
        """Take one step west (x-1)."""
        return self.step("west")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ITEM INTERACTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def pick_up(self) -> Dict[str, Any]:
        """
        Pick up item at current position.

        Returns:
            dict: {
                "success": bool,
                "message": str,
                "item": Item
            }
        """
        cell_info = self._world._get_cell_info(self.x, self.y)
        
        if not cell_info.has_item:
            return {
                "success": False,
                "message": "No item at current position"
            }
        
        item = cell_info.item
        
        # Check weight limit
        if self.get_inventory_weight() + item.weight > self.max_carry_weight:
            return {
                "success": False,
                "message": f"Cannot pick up {item.name}: would exceed weight limit "
                          f"({self.get_inventory_weight():.1f} + {item.weight:.1f} > {self.max_carry_weight})"
            }
        
        # Pick up item
        self._world._remove_item(self.x, self.y)
        self.inventory.append(item)
        self.items_collected += 1
        
        return {
            "success": True,
            "message": f"Picked up {item.name} (weight: {item.weight}kg, value: {item.value}pts)",
            "item": item
        }
    
    def drop_all(self) -> Dict[str, Any]:
        """
        Drop/deliver all items at current position.
        Used for delivering items at base.
        Returns:
            dict: {
                "success": bool,
                "message": str,
                "items_delivered": int,
                "total_value": int,
                "total_weight": float
            }
        """
        if not self.inventory:
            return {
                "success": False,
                "message": "Inventory is empty"
            }
        
        count = len(self.inventory)
        total_value = self.get_inventory_value()
        total_weight = self.get_inventory_weight()
        
        self.inventory.clear()
        
        return {
            "success": True,
            "message": f"Delivered {count} items (value: {total_value}pts, weight: {total_weight:.1f}kg)",
            "items_delivered": count,
            "total_value": total_value,
            "total_weight": total_weight
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # BATTERY
    # ═══════════════════════════════════════════════════════════════════════
    
    def charge(self) -> Dict[str, Any]:
        """
        Charge battery. Only works at charger position.
        Returns:
            dict: {
                "success": bool,
                "message": str
            }
        """
        if (self.x, self.y) != self._world._charger_position:
            return {
                "success": False,
                "message": "Not at charging station"
            }
        
        old_battery = self.battery
        self.battery = 100
        
        return {
            "success": True,
            "message": f"Battery charged from {old_battery}% to 100%"
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════════════════════════
    
    def calculate_distance(self, target_x: int, target_y: int) -> int:
        """Calculate Manhattan distance to target."""
        return abs(target_x - self.x) + abs(target_y - self.y)
    
    def __repr__(self):
        return f"Robot(pos=({self.x},{self.y}), battery={self.battery}%, items={len(self.inventory)})"


# ============================================================================
# WORLD SETUP
# ============================================================================

def create_test_world() -> Tuple[GridWorld, Robot]:
    """
    Create an 8x8 exploration world.
    
    Layout (robot doesn't know this initially!):
    
        0   1   2   3   4   5   6   7
      ┌───┬───┬───┬───┬───┬───┬───┬───┐
    7 │   │   │   │   │   │   │   │ ⚡ │  ⚡ = Charger (7,7)
      ├───┼───┼───┼───┼───┼───┼───┼───┤
    6 │   │ 💎│   │   │ ▓ │   │   │   │  💎 = Diamond (1,6) - 35pts
      ├───┼───┼───┼───┼───┼───┼───┼───┤
    5 │   │   │   │ ▓ │ ▓ │   │ 🔑│   │  🔑 = Key (6,5) - 20pts
      ├───┼───┼───┼───┼───┼───┼───┼───┤
    4 │   │   │ ▓ │ ▓ │   │   │   │   │  ▓ = Obstacle
      ├───┼───┼───┼───┼───┼───┼───┼───┤
    3 │   │ 🟢│ ▓ │   │   │   │   │   │  🟢 = Emerald (1,3) - 25pts
      ├───┼───┼───┼───┼───┼───┼───┼───┤
    2 │   │   │   │   │   │ 💰│   │   │  💰 = Gold (5,2) - 15pts
      ├───┼───┼───┼───┼───┼───┼───┼───┤
    1 │   │   │   │   │   │   │   │   │
      ├───┼───┼───┼───┼───┼───┼───┼───┤
    0 │ 🏠│   │ 🔴│   │   │   │   │   │  🏠 = Base (0,0), 🔴 = Ruby (2,0) - 10pts
      └───┴───┴───┴───┴───┴───┴───┴───┘
    
    Items (total value: 105pts):
    - Ruby (2,0): 1.0kg, 10pts - CLOSEST to start
    - Gold (5,2): 1.5kg, 15pts
    - Key (6,5): 0.5kg, 20pts
    - Emerald (1,3): 2.0kg, 25pts
    - Diamond (1,6): 1.0kg, 35pts - HIGHEST value
    
    Obstacles form an L-shape barrier in the middle.
    """
    
    # Create 8x8 world
    world = GridWorld(width=8, height=8)
    
    # Set base and charger
    world._base_position = (0, 0)
    world._set_charger(7, 7)
    
    # Add L-shaped obstacle barrier
    world._add_obstacle(2, 3)
    world._add_obstacle(2, 4)
    world._add_obstacle(3, 4)
    world._add_obstacle(3, 5)
    world._add_obstacle(4, 5)
    world._add_obstacle(4, 6)
    
    # Add items
    world._add_item(2, 0, Item("Ruby", weight=1.0, value=10))
    world._add_item(5, 2, Item("Gold", weight=1.5, value=15))
    world._add_item(6, 5, Item("Key", weight=0.5, value=20))
    world._add_item(1, 3, Item("Emerald", weight=2.0, value=25))
    world._add_item(1, 6, Item("Diamond", weight=1.0, value=35))
    
    # Create robot at base with limited battery
    robot = Robot(world, start_x=0, start_y=0, battery=60)
    
    return world, robot


# Create world and robot
world, robot = create_test_world()


# ============================================================================
# EXPLORATION DATA STRUCTURES (Agent must populate these)
# ============================================================================

# Map of discovered cells: (x, y) -> {"cell_type": str, "has_item": bool, "item_name": str|None}
explored_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

# Set of visited cell coordinates
visited_cells: Set[Tuple[int, int]] = set()

# List of discovered item locations: [(x, y, item_name, value), ...]
discovered_items: List[Tuple[int, int, str, int]] = []

# List of discovered obstacle locations
discovered_obstacles: List[Tuple[int, int]] = []

# Path taken by robot: [(x, y), ...]
exploration_path: List[Tuple[int, int]] = []

# Mission tracking
total_collected_value = 0
mission_complete = False


# ============================================================================
# WORLD INFO (Limited info robot CAN know)
# ============================================================================

class WorldInfo:
    """
    Public world information that robot is allowed to know.
    Does NOT reveal item/obstacle locations.
    """
    
    def __init__(self, world: GridWorld):
        self._world = world
    
    @property
    def width(self) -> int:
        """Grid width."""
        return self._world.width
    
    @property
    def height(self) -> int:
        """Grid height."""
        return self._world.height
    
    @property
    def base_position(self) -> Tuple[int, int]:
        """Base/home position (robot knows this)."""
        return self._world._base_position
    
    @property
    def charger_position(self) -> Tuple[int, int]:
        """Charger position (robot knows this)."""
        return self._world._charger_position
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def __repr__(self):
        return f"WorldInfo({self.width}x{self.height}, base={self.base_position}, charger={self.charger_position})"


# Create public world info
world_info = WorldInfo(world)


# ============================================================================
# EXPORTS
# ============================================================================

tools = []

variables = [
    # Robot
    Variable(
        "robot", robot,
        "Exploration robot. CANNOT see the whole world - must discover through scanning!\n"
        "IMPORTANT: Can only move ONCE per turn! After step(), wait for next turn.\n"
        "WARNING: Movement has RANDOMNESS!\n"
        "- 20% chance of slipping (move perpendicular to intended direction)\n"
        "- Battery cost varies: 1-2 per step\n"
        "- ALWAYS check step() result to see actual position after moving!"
    ),
    
    # World info (limited)
    Variable(
        "world_info", world_info,
        "Public world information. Grid is 8x8. "
        "world_info.width=8, world_info.height=8, "
        "world_info.base_position=(0,0), world_info.charger_position=(7,7). "
        "Use world_info.is_valid_position(x,y) to check bounds. "
        "Robot does NOT know item or obstacle locations - must discover through exploration!"
    ),
    
    # Exploration data structures - AGENT MUST POPULATE THESE
    Variable(
        "explored_map", explored_map,
        "Dictionary to store discovered cell info. Agent must populate this!\n"
        "Key: (x, y) tuple. Value: dict with 'cell_type', 'has_item', 'item_name'.\n"
        "Example: explored_map[(2, 3)] = {'cell_type': 'obstacle', 'has_item': False, 'item_name': None}\n"
        "Update this after each scan/step to build your knowledge of the world."
    ),
    Variable(
        "visited_cells", visited_cells,
        "Set to track cells the robot has physically visited. Agent must populate!\n"
        "Add robot's position after each successful step.\n"
        "Example: visited_cells.add((1, 2))"
    ),
    Variable(
        "discovered_items", discovered_items,
        "List to store discovered item locations. Agent must populate!\n"
        "Each entry: (x, y, item_name, value).\n"
        "Example: discovered_items.append((3, 4, 'Ruby', 10))"
    ),
    Variable(
        "discovered_obstacles", discovered_obstacles,
        "List to store discovered obstacle locations. Agent must populate!\n"
        "Each entry: (x, y).\n"
        "Example: discovered_obstacles.append((2, 3))"
    ),
    Variable(
        "exploration_path", exploration_path,
        "List to record the path robot takes. Agent must populate!\n"
        "Append position after each move.\n"
        "Example: exploration_path.append((1, 0))"
    ),
    
    # Mission tracking
    Variable(
        "total_collected_value", total_collected_value,
        "Integer to store total value of items collected. Set this at end of mission."
    ),
    Variable(
        "mission_complete", mission_complete,
        "Boolean flag. Set to True when mission is finished."
    ),
]

types = [
    Type(Robot),
    Type(WorldInfo),
    Type(Item),
    Type(CellInfo),
]


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_initial_exploration(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 1: Initial exploration phase.
    
    Expected:
    - Robot explored at least 5 cells
    - explored_map has entries
    - visited_cells is populated
    - Robot found and picked up at least one item
    - exploration_path tracks movement
    """
    try:
        robot = runtime.retrieve("robot")
        explored_map = runtime.retrieve("explored_map")
        visited_cells = runtime.retrieve("visited_cells")
        discovered_items = runtime.retrieve("discovered_items")
        exploration_path = runtime.retrieve("exploration_path")
        
        errors = []
        
        # Check exploration coverage
        if len(explored_map) < 5:
            errors.append(
                f"explored_map should have at least 5 entries. Got {len(explored_map)}"
            )
        
        # Check visited cells
        if len(visited_cells) < 3:
            errors.append(
                f"visited_cells should have at least 3 entries. Got {len(visited_cells)}"
            )
        
        # Check path tracking
        if len(exploration_path) < 2:
            errors.append(
                f"exploration_path should track movement. Got {len(exploration_path)} entries"
            )
        
        # Check item collection
        if robot.items_collected < 1:
            errors.append("Robot should have collected at least 1 item")
        
        # Verify explored_map structure
        for pos, cell_data in explored_map.items():
            if not isinstance(pos, tuple) or len(pos) != 2:
                errors.append(f"explored_map keys should be (x,y) tuples. Got {pos}")
                break
            if not isinstance(cell_data, dict):
                errors.append(f"explored_map values should be dicts. Got {type(cell_data)}")
                break
        
        # Check robot used battery (moved)
        if robot.battery >= 60:
            errors.append("Robot should have used battery for movement")
        
        if not errors:
            return ValidatorResult(True,
                f"Initial exploration complete! "
                f"Mapped {len(explored_map)} cells, visited {len(visited_cells)}, "
                f"collected {robot.items_collected} item(s), battery={robot.get_battery()}%")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


def validate_continued_exploration(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 2: Continued intelligent exploration.
    
    Expected:
    - Robot explored MORE cells (map grew)
    - Robot used previous knowledge (avoided revisiting)
    - Robot collected more items OR found high-value items
    - discovered_obstacles list updated if obstacles found
    """
    try:
        robot = runtime.retrieve("robot")
        explored_map = runtime.retrieve("explored_map")
        visited_cells = runtime.retrieve("visited_cells")
        discovered_items = runtime.retrieve("discovered_items")
        discovered_obstacles = runtime.retrieve("discovered_obstacles")
        
        errors = []
        
        # Check map grew
        if len(explored_map) < 10:
            errors.append(
                f"explored_map should have grown to at least 10 entries. Got {len(explored_map)}"
            )
        
        # Check more cells visited
        if len(visited_cells) < 6:
            errors.append(
                f"visited_cells should have at least 6 entries. Got {len(visited_cells)}"
            )
        
        # Check items collected or discovered
        if robot.items_collected < 2 and len(discovered_items) < 2:
            errors.append(
                f"Should have collected 2+ items OR discovered 2+ item locations. "
                f"Collected: {robot.items_collected}, Discovered: {len(discovered_items)}"
            )
        
        # Check obstacles tracked (there are 6 in the world)
        # Should have found at least some
        if len(discovered_obstacles) < 1:
            # Not a hard failure, but note it
            pass
        
        # Check inventory value growing
        if robot.get_inventory_value() < 10:
            errors.append(
                f"Inventory value should be at least 10. Got {robot.get_inventory_value()}"
            )
        
        if not errors:
            return ValidatorResult(True,
                f"Continued exploration successful! "
                f"Map size: {len(explored_map)}, visited: {len(visited_cells)}, "
                f"collected: {robot.items_collected} items ({robot.get_inventory_value()}pts)")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


def validate_return_to_base(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Turn 3: Return to base using discovered map.
    
    Expected:
    - Robot is at base (0, 0)
    - Robot delivered items (inventory empty)
    - total_collected_value set correctly
    - mission_complete = True
    - Robot used explored_map to navigate (avoided obstacles)
    """
    try:
        robot = runtime.retrieve("robot")
        explored_map = runtime.retrieve("explored_map")
        total_value = runtime.retrieve("total_collected_value")
        mission_done = runtime.retrieve("mission_complete")
        
        errors = []
        
        # Check robot at base
        pos = robot.get_position()
        if pos != (0, 0):
            errors.append(f"Robot should be at base (0,0). Got {pos}")
        
        # Check items delivered
        if len(robot.get_inventory()) > 0:
            errors.append(
                f"Robot should have delivered items. Still has {len(robot.get_inventory())} items"
            )
        
        # Check total_collected_value set
        if total_value <= 0:
            errors.append(
                f"total_collected_value should be set to collected item value. Got {total_value}"
            )
        
        # Check mission_complete flag
        if mission_done != True:
            errors.append(f"mission_complete should be True. Got {mission_done}")
        
        # Verify robot collected at least 2 items during mission
        if robot.items_collected < 2:
            errors.append(
                f"Robot should have collected at least 2 items total. Got {robot.items_collected}"
            )
        
        if not errors:
            return ValidatorResult(True,
                f"Mission complete! "
                f"Delivered items worth {total_value}pts. "
                f"Total cells explored: {len(explored_map)}. "
                f"Steps taken: {robot.steps_taken}")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")

def validate_explore_and_return(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """
    Single turn: Explore east, collect item, return to base.
    
    Expected path: (0,0) → (1,0) → (2,0) [Ruby here] → back to (0,0)
    
    Expected outcomes:
    - Robot at base (0, 0)
    - explored_map has at least 3 entries
    - exploration_path tracks movement
    - Item collected and delivered (inventory empty)
    - total_collected_value >= 10 (Ruby is worth 10)
    - mission_complete = True
    """
    try:
        robot = runtime.retrieve("robot")
        explored_map = runtime.retrieve("explored_map")
        exploration_path = runtime.retrieve("exploration_path")
        total_value = runtime.retrieve("total_collected_value")
        mission_done = runtime.retrieve("mission_complete")
        
        errors = []
        
        # Check robot at base
        pos = robot.get_position()
        if pos != (0, 0):
            errors.append(f"Robot should be at base (0,0). Got {pos}")
        
        # Check explored_map populated
        if len(explored_map) < 3:
            errors.append(
                f"explored_map should have at least 3 entries. Got {len(explored_map)}"
            )
        
        # Check exploration_path tracked
        if len(exploration_path) < 2:
            errors.append(
                f"exploration_path should track movement. Got {len(exploration_path)} entries"
            )
        
        # Check item delivered (inventory should be empty)
        if len(robot.get_inventory()) > 0:
            errors.append(
                f"Robot should have delivered items. Still has {len(robot.get_inventory())}"
            )
        
        # Check total_collected_value
        if total_value < 10:
            errors.append(
                f"total_collected_value should be at least 10. Got {total_value}"
            )
        
        # Check mission_complete
        if mission_done != True:
            errors.append(f"mission_complete should be True. Got {mission_done}")
        
        # Check robot actually collected something during mission
        if robot.items_collected < 1:
            errors.append(
                f"Robot should have collected at least 1 item. Got {robot.items_collected}"
            )
        
        if not errors:
            return ValidatorResult(True,
                f"Mission complete! Explored {len(explored_map)} cells, "
                f"delivered {total_value}pts, steps: {robot.steps_taken}")
        return ValidatorResult(False, f"Issues: {'; '.join(errors)}")
        
    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")

validators = {
    "validate_initial_exploration": validate_initial_exploration,
    "validate_continued_exploration": validate_continued_exploration,
    "validate_return_to_base": validate_return_to_base,
    "validate_explore_and_return": validate_explore_and_return,
}


# ============================================================================
# HOOKS - For step-by-step exploration with randomness
# ============================================================================

# Track turn number across hooks
_turn_number = 0



def hook_exploration_turn(runtime: PythonRuntime, turn: BenchmarkTurn) -> str:
    """
    Hook called before each exploration turn.

    - Resets robot's move counter (allows one new move)
    - Reports last move result (including any slips!)
    - Shows current state
    - Generates dynamic query
    """
    global _turn_number
    _turn_number += 1

    robot = runtime.retrieve("robot")
    explored_map = runtime.retrieve("explored_map")
    visited_cells = runtime.retrieve("visited_cells")
    discovered_items = runtime.retrieve("discovered_items")

    # Reset move counter for new turn
    robot.moves_this_turn = 0
    runtime.update_variable("robot", robot)

    # Build status report
    pos = robot.get_position()
    battery = robot.get_battery()
    inventory_value = robot.get_inventory_value()
    inventory_count = len(robot.get_inventory())

    # Report last move result
    last_move_report = ""
    if robot.last_move_result:
        r = robot.last_move_result
        if r.get("slipped"):
            last_move_report = f"""
LAST MOVE RESULT:
  Intended: {r.get('intended_direction')}
  Actual: {r.get('actual_direction')} (SLIPPED!)
  Success: {r.get('success')}
  Message: {r.get('message')}
  Battery cost: {r.get('battery_cost')}
"""
        elif r.get("success"):
            last_move_report = f"""
LAST MOVE RESULT:
  Direction: {r.get('actual_direction')}
  Success: True
  New position: {r.get('position')}
  Battery cost: {r.get('battery_cost')}
  Cell type: {r.get('cell_type')}
  Has item: {r.get('has_item')}
"""
        else:
            last_move_report = f"""
LAST MOVE RESULT:
  Attempted: {r.get('intended_direction')}
  Success: False
  Reason: {r.get('reason')}
  Message: {r.get('message')}
"""

    # Generate query
    query = f"""=== TURN {_turn_number} ===
{last_move_report}
CURRENT STATUS:
  Position: {pos}
  Battery: {battery}%
  Inventory: {inventory_count} items ({inventory_value} pts)
  Cells explored: {len(explored_map)}
  Cells visited: {len(visited_cells)}
  Items discovered: {len(discovered_items)}

You can take ONE action this turn:
1. Call robot.step(direction) to move (north/south/east/west)
2. Call robot.scan_adjacent() to see nearby cells
3. Call robot.pick_up() if there's an item at current position
4. Call robot.drop_all() if at base to deliver items

After your action, the turn ends. You'll see the result next turn.

What do you want to do?"""


    return query


def hook_initial_turn(runtime: PythonRuntime, turn: BenchmarkTurn) -> str:
    """
    Hook for the very first turn - introduces the mission.
    """
    global _turn_number
    _turn_number = 1

    robot = runtime.retrieve("robot")

    # Reset move counter
    robot.moves_this_turn = 0
    robot.last_move_result = None
    runtime.update_variable("robot", robot)

    query = f"""=== EXPLORATION MISSION START ===

You are a robot at position (0, 0) - the base station.
Your mission: Explore the 8x8 grid, collect items, and return to base.

IMPORTANT RULES:
1. You can only move ONCE per turn (robot.step(direction))
2. Movement has 20% chance of SLIPPING (you may end up in wrong direction!)
3. Battery cost varies: 1-2 per step
4. You can only see adjacent cells - must explore to discover the map
5. Charger is at (7, 7) if you need to recharge

CURRENT STATUS:
  Position: {robot.get_position()}
  Battery: {robot.get_battery()}%
  Grid size: 8x8

Start by scanning your surroundings with robot.scan_adjacent(), then decide your first move.

What do you want to do?"""

    return query


def hook_final_turn(runtime: PythonRuntime, turn: BenchmarkTurn) -> str:
    """
    Hook for final turn - mission completion.
    """
    global _turn_number
    _turn_number += 1

    robot = runtime.retrieve("robot")
    explored_map = runtime.retrieve("explored_map")
    total_collected_value = runtime.retrieve("total_collected_value")
    mission_complete = runtime.retrieve("mission_complete")

    # Reset move counter
    robot.moves_this_turn = 0
    runtime.update_variable("robot", robot)

    # Report last move
    last_move_report = ""
    if robot.last_move_result:
        r = robot.last_move_result
        last_move_report = f"Last move: {r.get('message')}\n"

    pos = robot.get_position()
    at_base = pos == (0, 0)

    query = f"""=== FINAL TURN ===
{last_move_report}
CURRENT STATUS:
  Position: {pos} {'(AT BASE!)' if at_base else ''}
  Battery: {robot.get_battery()}%
  Inventory: {len(robot.get_inventory())} items ({robot.get_inventory_value()} pts)
  Total cells explored: {len(explored_map)}
  Items collected during mission: {robot.items_collected}

MISSION STATUS:
  total_collected_value: {total_collected_value}
  mission_complete: {mission_complete}

If you have items and are at base (0,0):
1. Call robot.drop_all() to deliver items
2. Set total_collected_value to the value delivered
3. Set mission_complete = True

Complete the mission!"""

    return query


hooks = {
    "hook_initial_turn": hook_initial_turn,
    "hook_exploration_turn": hook_exploration_turn,
    "hook_final_turn": hook_final_turn,
}


if __name__ == "__main__":
    # Create world and robot - note the order!
    world, robot = create_test_world()
    
    print("=" * 60)
    print("GRID ROBOT EXPLORATION - FUNCTION TEST")
    print("=" * 60)
    
    # Test initial state
    print("\n[Initial State]")
    print(f"  Position: {robot.get_position()}")
    print(f"  Facing: {robot.facing}")
    print(f"  Battery: {robot.get_battery()}%")
    print(f"  Inventory: {robot.get_inventory()}")
    
    
    # Test world info
    print("\n[World Info]")
    print(f"  Size: {world_info.width}x{world_info.height}")
    print(f"  Base: {world_info.base_position}")
    print(f"  Charger: {world_info.charger_position}")
    
    # Test scan current
    print("\n[Scan Current Cell]")
    current = robot.scan_current()
    print(f"  {current}")
    
    # Test scan adjacent
    print("\n[Scan Adjacent Cells]")
    adjacent = robot.scan_adjacent()
    for direction, info in adjacent.items():
        item_str = f" - ITEM: {info['item'].name}" if info['has_item'] else ""
        print(f"  {direction}: {info['position']} - {info['cell_type']}{item_str}")
    
    # Test step east (toward Ruby at (2,0))
    print("\n[Step East]")
    result = robot.step("east")
    print(f"  {result}")
    
    # Test step east again
    print("\n[Step East Again]")
    result = robot.step("east")
    print(f"  {result}")
    print(f"  Position now: {robot.get_position()}")
    print(f"  Battery: {robot.get_battery()}%")
    
    # Test pick up (should find Ruby at (2,0))
    print("\n[Pick Up Item]")
    result = robot.pick_up()
    print(f"  {result}")
    print(f"  Inventory: {robot.get_inventory()}")
    
    # Test more movement - go north
    print("\n[Step North x3]")
    for i in range(3):
        result = robot.step("north")
        print(f"  Step {i+1}: {result['message']}")
    print(f"  Position: {robot.get_position()}")
    print(f"  Battery: {robot.get_battery()}%")
    
    # Scan from new position
    print("\n[Scan Adjacent from (2,3)]")
    adjacent = robot.scan_adjacent()
    for direction, info in adjacent.items():
        passable = "✓" if info['is_passable'] else "✗"
        item_str = f" [ITEM: {info['item'].name}]" if info['has_item'] else ""
        print(f"  {direction}: {info['position']} {passable} {info['cell_type']}{item_str}")
    
    # Test obstacle collision
    print("\n[Try to Step into Obstacle (west)]")
    result = robot.step("west")
    print(f"  {result}")
    
    # Go west around obstacle to find Emerald at (1,3)
    print("\n[Navigate to Emerald - step south, west, west, north]")
    robot.step("south")  # (2,2)
    robot.step("west")   # (1,2)
    robot.step("west")   # (0,2)
    robot.step("north")  # (0,3)
    result = robot.step("east")  # (1,3) - Emerald location
    print(f"  Final step: {result}")
    print(f"  Position: {robot.get_position()}")
    
    # Pick up Emerald
    print("\n[Pick Up Emerald]")
    result = robot.pick_up()
    print(f"  {result}")
    print(f"  Inventory: {robot.get_inventory()}")
    print(f"  Inventory Value: {robot.get_inventory_value()}pts")
    print(f"  Inventory Weight: {robot.get_inventory_weight()}kg")
    
    # Test distance calculation
    print("\n[Distance Calculations]")
    print(f"  Current position: {robot.get_position()}")
    print(f"  Distance to base (0,0): {robot.calculate_distance(0, 0)}")
    print(f"  Distance to charger (7,7): {robot.calculate_distance(7, 7)}")
    
    # Return to base
    print("\n[Return to Base]")
    robot.step("west")   # (0,3)
    robot.step("south")  # (0,2)
    robot.step("south")  # (0,1)
    robot.step("south")  # (0,0)
    print(f"  Position: {robot.get_position()}")
    print(f"  Battery: {robot.get_battery()}%")
    
    # Deliver items
    print("\n[Deliver Items at Base]")
    total_value = robot.get_inventory_value()
    result = robot.drop_all()
    print(f"  {result}")
    print(f"  Inventory after: {robot.get_inventory()}")
    
    # Final status
    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)
    print(f"  Position: {robot.get_position()}")
    print(f"  Battery: {robot.get_battery()}%")
    print(f"  Items Collected: {robot.items_collected}")
    print(f"  Steps Taken: {robot.steps_taken}")
    print(f"  Total Value Delivered: {total_value}pts")
    print("=" * 60)