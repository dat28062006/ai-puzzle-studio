from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import random
from time import perf_counter

from .engine import Level, PRESETS, State, corner_deadlock
from .solver import SearchResult, solve

TEMPLATES = {
    "Kho mở": (6, 6, frozenset()),
    "Trạm giữa": (7, 7, frozenset({(3, 3)})),
    "Lối vòng": (7, 7, frozenset({(2, 3), (4, 3)})),
}


@dataclass(frozen=True)
class Generated:
    level: Level
    verification: SearchResult
    attempts: int
    elapsed_ms: float
    note: str = ""


def generate(seed: int, *, boxes: int = 2, template: str = "Kho mở",
             max_attempts: int = 40, budget: float = 2.5) -> Generated:
    if boxes not in (1, 2):
        raise ValueError("Tạo màn hỗ trợ 1 hoặc 2 thùng.")
    if template not in TEMPLATES:
        raise ValueError("Khung bản đồ không hợp lệ.")
    rng = random.Random(seed)
    rows, cols, obstacles = TEMPLATES[template]
    walls = frozenset({(r, c) for r in range(rows) for c in range(cols)
                       if r in (0, rows - 1) or c in (0, cols - 1)} | set(obstacles))
    floor = sorted((r, c) for r in range(rows) for c in range(cols) if (r, c) not in walls)
    started = perf_counter()
    best: tuple[Level, SearchResult] | None = None
    attempts = 0
    for _ in range(max_attempts):
        remaining = budget - (perf_counter() - started)
        if remaining <= 0:
            break
        attempts += 1
        positions = rng.sample(floor, 2 * boxes + 1)
        goals = frozenset(positions[:boxes])
        crates = frozenset(positions[boxes:2 * boxes])
        state = State(positions[-1], crates)
        identity = hashlib.sha256(f"{template}:{seed}:{positions}".encode()).hexdigest()[:8]
        level = Level(f"gen-{identity}", f"Màn tạo · {seed}", rows, cols, walls, goals, state,
                      "generated", seed)
        if corner_deadlock(level, state):
            continue
        result = solve(level, timeout=min(0.18, remaining), max_states=15_000)
        if result.solved and result.actions:
            if best is None or len(result.actions) > len(best[1].actions):
                best = level, result
            if len(result.actions) >= (6 if boxes == 1 else 10):
                break
    note = ""
    if best is None:
        level = replace(PRESETS[0 if boxes == 1 else 1], source="preset_fallback", seed=seed)
        result = solve(level)
        note = "Chưa tạo được màn trong giới hạn; đang dùng màn mẫu đã kiểm tra."
    else:
        level, result = best
    return Generated(level, result, attempts, (perf_counter() - started) * 1000, note)
