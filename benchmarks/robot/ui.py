"""
Robot Exploration Game UI

A terminal-based game UI for visualizing robot exploration.
Uses Rich library for rendering.
"""

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich.box import ROUNDED, DOUBLE
from rich.columns import Columns
from typing import Dict, List, Any, Optional, Set, Tuple


class RobotGameUI:
    """Terminal-based game UI for robot exploration."""

    def __init__(self, width: int = 8, height: int = 8):
        self.console = Console()
        self.width = width
        self.height = height
        self.live: Optional[Live] = None

        # State
        self.robot_pos: Tuple[int, int] = (0, 0)
        self.robot_facing: str = "north"
        self.robot_battery: int = 60
        self.robot_inventory: List[Dict] = []

        # World knowledge
        self.explored_cells: Dict[Tuple[int, int], Dict] = {}
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.discovered_items: List[Tuple[int, int, str, int]] = []
        self.discovered_obstacles: Set[Tuple[int, int]] = set()

        # Fixed locations
        self.base_pos = (0, 0)
        self.charger_pos = (7, 7)

        # Turn info
        self.turn_number: int = 0
        self.last_action: str = ""
        self.last_result: Dict = {}
        self.action_log: List[str] = []

        # Stats
        self.total_steps: int = 0
        self.items_collected: int = 0
        self.total_value: int = 0
        self.mission_complete: bool = False

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    def start(self):
        """Start the live display."""
        if self.live is not None:
            return
        self.live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=4,
            screen=True,
            transient=False,
        )
        self.live.start()

    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.update(self._build_layout())
            self.live.stop()
            self.live = None

    def refresh(self):
        """Refresh the display."""
        if self.live:
            self.live.update(self._build_layout())
        else:
            self.render()

    def render(self):
        """Simple render without Live display."""
        self.console.clear()
        self.console.print(self._build_layout())

    # ========================================================================
    # UPDATE METHODS
    # ========================================================================

    def update_robot(self, position: Tuple[int, int], battery: int,
                     facing: str = "north", inventory: List = None):
        self.robot_pos = position
        self.robot_battery = battery
        self.robot_facing = facing
        if inventory is not None:
            self.robot_inventory = inventory

    def update_explored(self, explored_map: Dict[Tuple[int, int], Dict]):
        self.explored_cells = dict(explored_map) if explored_map else {}

    def update_visited(self, visited: Set[Tuple[int, int]]):
        self.visited_cells = set(visited) if visited else set()

    def update_discovered_items(self, items: List[Tuple[int, int, str, int]]):
        self.discovered_items = list(items) if items else []

    def update_obstacles(self, obstacles: Set[Tuple[int, int]]):
        self.discovered_obstacles = set(obstacles) if obstacles else set()

    def set_turn(self, turn: int):
        self.turn_number = turn

    def set_last_action(self, action: str, result: Dict = None):
        self.last_action = action
        self.last_result = result or {}

    def add_action_log(self, message: str):
        self.action_log.append(message)
        if len(self.action_log) > 10:
            self.action_log = self.action_log[-10:]

    def set_stats(self, steps: int = 0, items: int = 0, value: int = 0,
                  complete: bool = False):
        self.total_steps = steps
        self.items_collected = items
        self.total_value = value
        self.mission_complete = complete

    # ========================================================================
    # GRID RENDERING
    # ========================================================================

    def _get_cell_char(self, x: int, y: int) -> Tuple[str, str]:
        """Get character and color for a cell."""
        pos = (x, y)

        if pos == self.robot_pos:
            return "@", "bold yellow"
        if pos == self.base_pos:
            return "H", "bold green"
        if pos == self.charger_pos:
            return "+", "bold bright_yellow"

        if pos in self.explored_cells:
            cell_data = self.explored_cells[pos]

            if cell_data.get('cell_type') == 'obstacle':
                return "#", "bold red"

            if cell_data.get('has_item'):
                item_name = (cell_data.get('item_name') or '').lower()
                if 'ruby' in item_name:
                    return "R", "bold bright_red"
                elif 'emerald' in item_name:
                    return "E", "bold bright_green"
                elif 'gold' in item_name:
                    return "G", "bold bright_yellow"
                elif 'diamond' in item_name:
                    return "D", "bold bright_cyan"
                elif 'key' in item_name:
                    return "K", "bold bright_magenta"
                else:
                    return "*", "bold cyan"

            if pos in self.visited_cells:
                return ".", "blue"

            return ".", "dim"

        return " ", "dim"

    def _build_grid(self) -> Table:
        """Build the game grid as a table."""
        grid = Table(
            show_header=True,
            header_style="bold cyan",
            box=ROUNDED,
            padding=(0, 1),
        )

        # Column headers
        grid.add_column("", style="bold cyan", justify="center", width=2)
        for x in range(self.width):
            grid.add_column(str(x), justify="center", width=2)

        # Rows (y from top to bottom)
        for y in range(self.height - 1, -1, -1):
            row = [f"{y}"]
            for x in range(self.width):
                char, color = self._get_cell_char(x, y)
                row.append(f"[{color}]{char}[/]")
            grid.add_row(*row)

        return grid

    # ========================================================================
    # PANEL RENDERING
    # ========================================================================

    def _build_robot_panel(self) -> Panel:
        """Build robot status panel."""
        battery_pct = max(0, min(100, self.robot_battery))
        bar_len = 10
        filled = int(battery_pct / 100 * bar_len)

        if battery_pct > 50:
            bar_color = "green"
        elif battery_pct > 25:
            bar_color = "yellow"
        else:
            bar_color = "red"

        battery_bar = f"[{bar_color}]{'█' * filled}[/][dim]{'░' * (bar_len - filled)}[/] {battery_pct}%"

        inv_count = len(self.robot_inventory)
        inv_value = sum(i.get('value', 0) for i in self.robot_inventory)

        lines = [
            f"Battery:   {battery_bar}",
            f"Position:  ({self.robot_pos[0]}, {self.robot_pos[1]})",
            f"Facing:    {self.robot_facing.capitalize()}",
            f"Inventory: {inv_count} items ({inv_value} pts)",
        ]

        return Panel(
            Text.from_markup("\n".join(lines)),
            title="[bold green]Robot[/]",
            border_style="green",
            box=ROUNDED,
        )

    def _build_mission_panel(self) -> Panel:
        """Build mission stats panel."""
        status = "[bold green]COMPLETE[/]" if self.mission_complete else "[dim]In Progress[/]"

        lines = [
            f"Turn:      [bold yellow]{self.turn_number}[/]",
            f"Steps:     {self.total_steps}",
            f"Explored:  {len(self.explored_cells)} cells",
            f"Visited:   {len(self.visited_cells)} cells",
            f"Collected: {self.items_collected} items",
            f"Value:     {self.total_value} pts",
            f"Status:    {status}",
        ]

        return Panel(
            Text.from_markup("\n".join(lines)),
            title="[bold yellow]Mission[/]",
            border_style="yellow",
            box=ROUNDED,
        )

    def _build_action_panel(self) -> Panel:
        """Build last action panel."""
        if not self.last_result:
            content = "[dim]Waiting for action...[/]"
        else:
            r = self.last_result
            lines = []

            success = r.get('success', False)
            icon = "[green]OK[/]" if success else "[red]FAIL[/]"
            msg = r.get('message', 'Unknown')
            lines.append(f"{icon} {msg}")

            if r.get('slipped'):
                lines.append("")
                lines.append(f"[bold magenta]SLIPPED![/]")
                lines.append(f"  Intended: {r.get('intended_direction')}")
                lines.append(f"  Actual:   {r.get('actual_direction')}")

            if r.get('battery_cost'):
                lines.append(f"[dim]Battery cost: {r.get('battery_cost')}[/]")

            content = "\n".join(lines)

        return Panel(
            Text.from_markup(content),
            title="[bold magenta]Last Action[/]",
            border_style="magenta",
            box=ROUNDED,
        )

    def _build_log_panel(self) -> Panel:
        """Build action log panel."""
        if not self.action_log:
            content = "[dim]No actions yet[/]"
        else:
            lines = []
            for i, msg in enumerate(self.action_log[-5:]):
                if i == len(self.action_log[-5:]) - 1:
                    lines.append(f"[bold]> {msg}[/]")
                else:
                    lines.append(f"[dim]> {msg}[/]")
            content = "\n".join(lines)

        return Panel(
            Text.from_markup(content),
            title="[bold blue]Log[/]",
            border_style="blue",
            box=ROUNDED,
        )

    def _build_legend(self) -> Text:
        """Build legend text."""
        return Text.from_markup(
            "[yellow]@[/]=Robot  [green]H[/]=Base  [bright_yellow]+[/]=Charger  "
            "[red]#[/]=Wall  [cyan]*[/]=Item  [blue].[/]=Visited  [dim] [/]=Unknown"
        )

    # ========================================================================
    # MAIN LAYOUT
    # ========================================================================

    def _build_layout(self) -> Panel:
        """Build complete game layout."""
        # Left side: grid
        grid_panel = Panel(
            Align.center(self._build_grid()),
            title="[bold cyan]Map[/]",
            border_style="cyan",
            box=ROUNDED,
        )

        # Right side: status panels
        right_side = Group(
            self._build_robot_panel(),
            self._build_mission_panel(),
            self._build_action_panel(),
        )

        # Top row: grid + status
        top_row = Columns([grid_panel, right_side], equal=False, expand=True)

        # Full layout
        layout = Group(
            top_row,
            self._build_log_panel(),
            Align.center(self._build_legend()),
        )

        return Panel(
            layout,
            title=f"[bold white on blue] Robot Exploration - Turn {self.turn_number} [/]",
            border_style="blue",
            box=DOUBLE,
            padding=(1, 2),
        )


# ============================================================================
# GLOBAL INSTANCE & HELPERS
# ============================================================================

_game_ui: Optional[RobotGameUI] = None


def get_game_ui() -> Optional[RobotGameUI]:
    return _game_ui


def create_game_ui(width: int = 8, height: int = 8) -> RobotGameUI:
    global _game_ui
    _game_ui = RobotGameUI(width=width, height=height)
    return _game_ui


def start_ui():
    global _game_ui
    if _game_ui is not None:
        _game_ui.start()


def stop_ui():
    global _game_ui
    if _game_ui is not None:
        _game_ui.stop()


def update_ui_from_runtime(runtime, turn_number: int = 0):
    global _game_ui
    if _game_ui is None:
        return

    try:
        robot = runtime.get_variable("robot")
        _game_ui.update_robot(
            position=robot.get_position(),
            battery=robot.get_battery(),
            facing=robot.facing,
            inventory=[{"name": i.name, "value": i.value} for i in robot.get_inventory()]
        )
        _game_ui.set_stats(steps=robot.steps_taken, items=robot.items_collected)
        if robot.last_move_result:
            _game_ui.set_last_action("move", robot.last_move_result)
            r = robot.last_move_result
            if r.get('slipped'):
                _game_ui.add_action_log(f"SLIP! {r.get('intended_direction')}->{r.get('actual_direction')}")
            elif r.get('success'):
                _game_ui.add_action_log(f"Moved {r.get('actual_direction')} to {r.get('position')}")
    except Exception:
        pass

    try:
        _game_ui.update_explored(runtime.get_variable("explored_map"))
    except Exception:
        pass

    try:
        _game_ui.update_visited(runtime.get_variable("visited_cells"))
    except Exception:
        pass

    try:
        _game_ui.update_discovered_items(runtime.get_variable("discovered_items"))
    except Exception:
        pass

    try:
        _game_ui.set_stats(
            steps=_game_ui.total_steps,
            items=_game_ui.items_collected,
            value=runtime.get_variable("total_collected_value"),
            complete=runtime.get_variable("mission_complete"),
        )
    except Exception:
        pass

    _game_ui.set_turn(turn_number)
    _game_ui.refresh()


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    import time

    ui = RobotGameUI(width=8, height=8)

    # Initial state
    ui.update_robot(position=(0, 0), battery=60, facing="north")
    ui.update_explored({(0, 0): {'cell_type': 'base', 'has_item': False}})
    ui.update_visited({(0, 0)})
    ui.set_turn(1)

    # Start Live display
    ui.start()
    time.sleep(1)

    # Simulate moves
    moves = [
        ((1, 0), 58, "east", False, "Moved east to (1, 0)"),
        ((2, 0), 56, "east", False, "Moved east to (2, 0)"),
        ((2, 0), 54, "east", True, "SLIPPED! Tried east, went north"),
        ((3, 0), 52, "east", False, "Moved east to (3, 0)"),
    ]

    explored = {(0, 0): {'cell_type': 'base', 'has_item': False}}
    visited = {(0, 0)}

    for i, (pos, battery, direction, slipped, log_msg) in enumerate(moves, 2):
        time.sleep(1)

        explored[pos] = {'cell_type': 'empty', 'has_item': i == 3, 'item_name': 'Ruby' if i == 3 else None}
        visited.add(pos)

        ui.update_robot(position=pos, battery=battery, facing=direction)
        ui.update_explored(explored)
        ui.update_visited(visited)
        ui.set_turn(i)
        ui.set_last_action("step", {
            'success': True,
            'message': f"Moved {direction} to {pos}",
            'slipped': slipped,
            'intended_direction': 'east' if slipped else direction,
            'actual_direction': direction,
            'battery_cost': 2,
        })
        ui.add_action_log(log_msg)
        ui.set_stats(steps=i - 1, items=1 if i > 3 else 0, value=10 if i > 3 else 0)
        ui.refresh()

    time.sleep(2)
    ui.stop()
    print("\nDemo complete!")
