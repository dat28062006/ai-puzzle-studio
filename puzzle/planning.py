"""Push actions and admissible lower bounds shared by search and the RL environment.

Walking BFS only finds a path to the side of a crate. It never chooses which
crate to push or provides an episode solution to the learning agent.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

from .engine import Cell, DIRECTIONS, Level, State, add


@dataclass(frozen=True)
class PushAction:
    index: int
    state: State
    actions: tuple[str, ...]


def walking_paths(level: Level, state: State) -> dict[Cell, tuple[str, ...]]:
    paths = {state.player: ()}
    queue = deque([state.player])
    while queue:
        current = queue.popleft()
        for action, delta in DIRECTIONS.items():
            target = add(current, delta)
            if target not in paths and target not in state.boxes and not level.blocked(target):
                paths[target] = paths[current] + (action,)
                queue.append(target)
    return paths


def push_actions(level: Level, state: State) -> tuple[PushAction, ...]:
    paths = walking_paths(level, state)
    result = []
    for box_index, box in enumerate(sorted(state.boxes)):
        for direction_index, (action, delta) in enumerate(DIRECTIONS.items()):
            stand = box[0] - delta[0], box[1] - delta[1]
            destination = add(box, delta)
            if stand in paths and not level.blocked(destination) and destination not in state.boxes:
                nxt = State(box, frozenset((state.boxes - {box}) | {destination}))
                result.append(PushAction(box_index * 4 + direction_index, nxt, paths[stand] + (action,)))
    return tuple(result)


@lru_cache(maxsize=128)
def reverse_distances(rows: int, cols: int, walls: frozenset, goals: frozenset) -> tuple:
    """Static box-to-goal distances, allowing only pushes with a standing square."""
    maps = []
    def floor(p):
        return 0 <= p[0] < rows and 0 <= p[1] < cols and p not in walls
    for goal in sorted(goals):
        distances = {goal: 0}
        queue = deque([goal])
        while queue:
            current = queue.popleft()
            for dr, dc in DIRECTIONS.values():
                previous = current[0] - dr, current[1] - dc
                stand = current[0] - 2 * dr, current[1] - 2 * dc
                if floor(previous) and floor(stand) and previous not in distances:
                    distances[previous] = distances[current] + 1
                    queue.append(previous)
        maps.append(distances)
    return tuple(maps)


def push_lower_bound(level: Level, state: State) -> float:
    distances = reverse_distances(level.rows, level.cols, level.walls, level.goals)
    crates = sorted(state.boxes)
    return min(sum(distances[g].get(box, float("inf")) for box, g in zip(crates, order))
               for order in permutations(range(len(distances))))
