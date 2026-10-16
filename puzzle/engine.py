from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

Cell = tuple[int, int]
DIRECTIONS: dict[str, Cell] = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
ARROWS = {"U": "↑", "D": "↓", "L": "←", "R": "→"}


@dataclass(frozen=True)
class State:
    player: Cell
    boxes: frozenset[Cell]

    def to_dict(self) -> dict:
        return {"player": list(self.player), "boxes": [list(p) for p in sorted(self.boxes)]}


@dataclass(frozen=True)
class Level:
    id: str
    title: str
    rows: int
    cols: int
    walls: frozenset[Cell]
    goals: frozenset[Cell]
    initial: State
    source: str = "preset"
    seed: int | None = None

    def inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def blocked(self, cell: Cell) -> bool:
        return not self.inside(cell) or cell in self.walls

    @property
    def floor(self) -> frozenset[Cell]:
        return frozenset((r, c) for r in range(self.rows) for c in range(self.cols)
                         if (r, c) not in self.walls)

    def to_dict(self) -> dict:
        return {"schema_version": 1, "id": self.id, "title": self.title,
                "source": self.source, "seed": self.seed,
                "grid": to_ascii(self).splitlines()}


@dataclass(frozen=True)
class Move:
    state: State
    valid: bool
    pushed: bool = False
    reason: str = ""


def add(a: Cell, b: Cell) -> Cell:
    return a[0] + b[0], a[1] + b[1]


def step(level: Level, state: State, action: str) -> Move:
    if action not in DIRECTIONS:
        raise ValueError(f"Hướng không hợp lệ: {action}")
    delta = DIRECTIONS[action]
    target = add(state.player, delta)
    if level.blocked(target):
        return Move(state, False, reason="Có tường ở phía trước.")
    if target in state.boxes:
        destination = add(target, delta)
        if level.blocked(destination) or destination in state.boxes:
            return Move(state, False, reason="Không đủ chỗ để đẩy thùng.")
        boxes = (state.boxes - {target}) | {destination}
        return Move(State(target, frozenset(boxes)), True, True)
    return Move(State(target, state.boxes), True)


def won(level: Level, state: State) -> bool:
    return state.boxes == level.goals


def corner_deadlock(level: Level, state: State) -> bool:
    for row, col in state.boxes - level.goals:
        vertical = level.blocked((row - 1, col)) or level.blocked((row + 1, col))
        horizontal = level.blocked((row, col - 1)) or level.blocked((row, col + 1))
        if vertical and horizontal:
            return True
    return False


def validate(level: Level) -> None:
    if not 3 <= level.rows <= 10 or not 3 <= level.cols <= 10:
        raise ValueError("Bản đồ cần có từ 3 đến 10 hàng/cột.")
    if not 1 <= len(level.initial.boxes) <= 3:
        raise ValueError("Demo hỗ trợ từ 1 đến 3 thùng.")
    if len(level.initial.boxes) != len(level.goals):
        raise ValueError("Số thùng phải bằng số đích.")
    objects = level.initial.boxes | level.goals | {level.initial.player}
    if any(level.blocked(p) for p in objects):
        raise ValueError("Nhân vật, thùng và đích phải nằm trên nền.")
    if level.initial.player in level.initial.boxes:
        raise ValueError("Nhân vật không thể trùng vị trí thùng.")
    boundary = {(r, c) for r in range(level.rows) for c in range(level.cols)
                if r in (0, level.rows - 1) or c in (0, level.cols - 1)}
    if not boundary <= level.walls:
        raise ValueError("Bản đồ phải có tường bao kín.")


def from_ascii(text: str | Iterable[str], *, id: str = "imported", title: str = "Màn nhập",
               source: str = "imported", seed: int | None = None) -> Level:
    lines = text.strip("\r\n").splitlines() if isinstance(text, str) else list(text)
    if not lines or not lines[0] or any(len(line) != len(lines[0]) for line in lines):
        raise ValueError("Bản đồ phải là hình chữ nhật.")
    walls, goals, boxes, players = set(), set(), set(), []
    for row, line in enumerate(lines):
        for col, char in enumerate(line):
            cell = (row, col)
            if char not in "# .@$*+":
                raise ValueError(f"Ký hiệu không hợp lệ: {char!r}")
            if char == "#": walls.add(cell)
            if char in ".*+": goals.add(cell)
            if char in "$*": boxes.add(cell)
            if char in "@+": players.append(cell)
    if len(players) != 1:
        raise ValueError("Bản đồ phải có đúng một nhân vật (@ hoặc +).")
    level = Level(str(id)[:80], str(title)[:100], len(lines), len(lines[0]),
                  frozenset(walls), frozenset(goals), State(players[0], frozenset(boxes)),
                  source, seed)
    validate(level)
    return level


def from_dict(data: dict) -> Level:
    if not isinstance(data, dict) or not isinstance(data.get("grid"), list):
        raise ValueError("JSON cần chứa danh sách dòng trong trường grid.")
    if len(data["grid"]) > 10 or any(not isinstance(s, str) or len(s) > 10 for s in data["grid"]):
        raise ValueError("Bản đồ nhập vượt kích thước hỗ trợ.")
    return from_ascii(data["grid"], id=data.get("id", "imported"),
                      title=data.get("title", "Màn nhập"), source="imported",
                      seed=data.get("seed") if isinstance(data.get("seed"), int) else None)


def to_ascii(level: Level, state: State | None = None) -> str:
    state = state or level.initial
    lines = []
    for row in range(level.rows):
        line = ""
        for col in range(level.cols):
            cell = (row, col)
            char = "#" if cell in level.walls else "." if cell in level.goals else " "
            if cell in state.boxes: char = "*" if cell in level.goals else "$"
            if cell == state.player: char = "+" if cell in level.goals else "@"
            line += char
        lines.append(line)
    return "\n".join(lines)


PRESETS = (
    from_ascii("######\n#    #\n# .  #\n# $  #\n# @  #\n######",
               id="first-push", title="01 · Cú đẩy đầu tiên", source="preset"),
    from_ascii("######\n# . .#\n# $ $#\n# @  #\n#    #\n######",
               id="double-orbit", title="02 · Quỹ đạo kép", source="preset"),
    from_ascii("#######\n# .   #\n#   # #\n# $$  #\n#   . #\n#  @  #\n#######",
               id="detour", title="03 · Đường vòng", source="preset"),
)
