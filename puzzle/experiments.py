"""Compare independent agents on exactly the same starting state."""
from .engine import Level, State
from .q_learning import load_model, rollout
from .solver import replay, solve


def compare_agents(level: Level, start: State | None = None, *, timeout: float = 3.0,
                   max_states: int = 60_000) -> list[dict]:
    start = start or level.initial
    results = [solve(level, start, timeout=timeout, max_states=max_states, algorithm=algorithm)
               for algorithm in ("bfs", "astar_moves", "astar_pushes")]
    results.append(rollout(level, load_model(level), start))
    rows = []
    for result in results:
        if result.solved:
            replay(level, result)
        rows.append({"Agent": result.algorithm, "Kết quả": result.status,
                     "Bước đi": len(result.actions) if result.solved else None,
                     "Lượt đẩy": result.pushes if result.solved else None,
                     "Trạng thái / quyết định": result.explored,
                     "Thời gian (ms)": round(result.elapsed_ms, 3),
                     "Bảo đảm tối ưu": result.optimal_for if result.solved and result.optimal_for else "Không",
                     "Lý do": result.reason})
    return rows
