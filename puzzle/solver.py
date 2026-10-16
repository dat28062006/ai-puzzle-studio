from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from heapq import heappop, heappush
from itertools import count

from .engine import DIRECTIONS, Level, State, corner_deadlock, step, won
from .planning import push_actions, push_lower_bound


@dataclass(frozen=True)
class SearchResult:
    status: str
    actions: tuple[str, ...]
    pushes: int
    explored: int
    discovered: int
    elapsed_ms: float
    start: State
    reason: str = ""
    algorithm: str = "Breadth-first search (BFS)"
    optimal_for: str | None = "player_moves"
    metadata: dict | None = None

    @property
    def solved(self) -> bool:
        return self.status == "solved"

    def to_dict(self) -> dict:
        return {"algorithm": self.algorithm, "status": self.status,
                "actions": list(self.actions), "moves": len(self.actions) if self.solved else None,
                "pushes": self.pushes if self.solved else None,
                "explored_states": self.explored, "discovered_states": self.discovered,
                "elapsed_ms": round(self.elapsed_ms, 3),
                "optimal_for": self.optimal_for if self.solved else None,
                "start_state": self.start.to_dict(), "reason": self.reason,
                "metadata": self.metadata}


def solve(level: Level, start: State | None = None, *, max_states: int = 60_000,
          timeout: float = 3.0, algorithm: str = "bfs") -> SearchResult:
    """Shortest player-move path on small boards; only sound corner pruning is used."""
    if algorithm in ("astar_moves", "astar_pushes"):
        return solve_astar(level, start, max_states=max_states, timeout=timeout, objective=algorithm)
    if algorithm != "bfs":
        raise ValueError("Thuật toán tìm kiếm không hợp lệ.")
    if max_states < 1:
        raise ValueError("max_states phải lớn hơn 0")
    start = start or level.initial
    begin = perf_counter()
    explored = 0
    parents: dict[State, tuple[State, str, bool] | None] = {start: None}

    def finish(status: str, terminal: State | None = None, reason: str = "") -> SearchResult:
        actions, pushes = [], 0
        while terminal is not None and parents[terminal] is not None:
            previous, action, pushed = parents[terminal]
            actions.append(action)
            pushes += int(pushed)
            terminal = previous
        return SearchResult(status, tuple(reversed(actions)), pushes, explored, len(parents),
                            (perf_counter() - begin) * 1000, start, reason)

    if won(level, start):
        return finish("solved", start, "Tất cả thùng đã ở trên đích.")
    if corner_deadlock(level, start):
        return finish("no_solution", reason="Một thùng bị kẹt ở góc không có đích.")
    queue = deque([start])
    while queue:
        if perf_counter() - begin >= timeout:
            return finish("limit_reached", reason="Đạt giới hạn thời gian; chưa thể kết luận vô nghiệm.")
        current = queue.popleft()
        explored += 1
        for action in DIRECTIONS:
            movement = step(level, current, action)
            nxt = movement.state
            if not movement.valid or nxt in parents or corner_deadlock(level, nxt):
                continue
            if len(parents) >= max_states:
                return finish("limit_reached", reason="Đạt giới hạn trạng thái; chưa thể kết luận vô nghiệm.")
            parents[nxt] = (current, action, movement.pushed)
            if won(level, nxt):
                return finish("solved", nxt)
            queue.append(nxt)
    return finish("no_solution", reason="Đã duyệt hết trạng thái hợp lệ mà không tìm thấy lời giải.")


def solve_astar(level: Level, start: State | None = None, *, max_states: int = 60_000,
                timeout: float = 3.0, objective: str = "astar_moves") -> SearchResult:
    """A* over pushes, with lexicographic optimality and exact shortest walks.

    Keep the actual player square in the state: merging reachable areas would
    lose the walking cost and invalidate the guarantee about player moves.
    """
    if max_states < 1 or objective not in ("astar_moves", "astar_pushes"):
        raise ValueError("Cấu hình A* không hợp lệ.")
    start = start or level.initial
    began = perf_counter()
    explored = 0
    best = {start: (0, 0)}
    parents = {start: None}
    order = count()
    moves_first = objective == "astar_moves"
    label = "A* push-macro (moves, pushes)" if moves_first else "A* push-macro (pushes, moves)"
    optimal_for = "player_moves_then_pushes" if moves_first else "pushes_then_player_moves"

    def finish(status, terminal=None, reason=""):
        parts = []
        while terminal is not None and parents[terminal] is not None:
            previous, actions = parents[terminal]
            parts.append(actions)
            terminal = previous
        actions = tuple(action for part in reversed(parts) for action in part)
        return SearchResult(status, actions, len(parts), explored, len(best),
                            (perf_counter() - began) * 1000, start, reason, label, optimal_for)

    if won(level, start):
        return finish("solved", start)
    bound = push_lower_bound(level, start)
    if bound == float("inf"):
        return finish("no_solution", reason="Có thùng không thể được đẩy đến một đích phù hợp.")
    heap = [(bound, bound, next(order), (0, 0), start)]
    while heap:
        if perf_counter() - began >= timeout:
            return finish("limit_reached", reason="Đạt giới hạn thời gian; chưa thể kết luận vô nghiệm.")
        _, _, _, cost, current = heappop(heap)
        if best.get(current) != cost:
            continue
        explored += 1
        if won(level, current):
            return finish("solved", current)
        for action in push_actions(level, current):
            nxt = action.state
            lower = push_lower_bound(level, nxt)
            if lower == float("inf"):
                continue
            edge = (len(action.actions), 1) if moves_first else (1, len(action.actions))
            candidate = cost[0] + edge[0], cost[1] + edge[1]
            if candidate >= best.get(nxt, (float("inf"), float("inf"))):
                continue
            if nxt not in best and len(best) >= max_states:
                return finish("limit_reached", reason="Đạt giới hạn trạng thái; chưa thể kết luận vô nghiệm.")
            best[nxt] = candidate
            parents[nxt] = current, action.actions
            heappush(heap, (candidate[0] + lower, candidate[1] + lower, next(order), candidate, nxt))
    return finish("no_solution", reason="Đã duyệt hết trạng thái đẩy hợp lệ mà không tìm thấy lời giải.")


def replay(level: Level, result: SearchResult) -> tuple[State, ...]:
    if not result.solved:
        return (result.start,)
    states = [result.start]
    for action in result.actions:
        movement = step(level, states[-1], action)
        if not movement.valid:
            raise ValueError("Lời giải chứa bước không hợp lệ.")
        states.append(movement.state)
    if not won(level, states[-1]):
        raise ValueError("Lời giải không đưa tất cả thùng đến đích.")
    return tuple(states)
