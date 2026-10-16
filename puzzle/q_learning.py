"""Actual model-free Q-learning; no search solver, demonstrations or fallback.

The policy is specific to a wall/goal layout. Its Q-table is learned from
environment rewards. Shortest walking paths implement a selected push action;
they do not decide the sequence of pushes needed to solve the puzzle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Callable

from .engine import DIRECTIONS, Level, State, corner_deadlock, won
from .planning import walking_paths
from .rl_env import SokobanPushEnv
from .solver import SearchResult

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def layout_key(level: Level) -> str:
    layout = (level.rows, level.cols, sorted(level.walls), sorted(level.goals))
    return hashlib.sha256(json.dumps(layout).encode()).hexdigest()[:20]


def state_key(state: State) -> str:
    cells = [state.player, *sorted(state.boxes)]
    return ";".join(f"{r},{c}" for r, c in cells)


def policy_key(level: Level, state: State) -> str:
    # Retain the exact player square because walking has a reward cost.
    return state_key(state)


@dataclass
class QModel:
    layout: str
    q: dict[str, dict[int, float]] = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def to_dict(self):
        return {"schema_version": 1, "algorithm": "tabular_q_learning_push_actions", "layout": self.layout,
                "q": self.q, "training": self.training, "history": self.history}

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version") != 1 or data.get("algorithm") != "tabular_q_learning_push_actions":
            raise ValueError("Checkpoint Q-learning không được hỗ trợ.")
        return cls(data["layout"], {key: {int(a): float(v) for a, v in row.items()} for key, row in data["q"].items()},
                   data.get("training", {}), data.get("history", []))


def checkpoint_path(level: Level) -> Path:
    return MODEL_DIR / f"q-{layout_key(level)}.json"


def save_model(level: Level, model: QModel) -> Path:
    if layout_key(level) != model.layout:
        raise ValueError("Policy không thuộc khung tường và đích này.")
    path = checkpoint_path(level)
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(model.to_dict(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)
    return path


def load_model(level: Level) -> QModel | None:
    path = checkpoint_path(level)
    if not path.exists():
        return None
    model = QModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if model.layout != layout_key(level):
        raise ValueError("Checkpoint không khớp bản đồ.")
    if model.training.get("state_abstraction") != "exact_player_and_sorted_boxes":
        return None
    return model


def rollout(level: Level, model: QModel | None, start: State | None = None, *, max_pushes: int = 80) -> SearchResult:
    start = start or level.initial
    began = perf_counter()
    actions, pushes, visited = [], 0, set()
    state = start

    def finish(status, reason=""):
        return SearchResult(status, tuple(actions), pushes, pushes, len(visited), (perf_counter() - began) * 1000,
                            start, reason, "RL · Tabular Q-learning (push actions)", None,
                            {"layout": layout_key(level), "training": model.training if model else None,
                             "inference_uses_search_solver": False, "attempted_player_moves": len(actions)})

    if model is None or not model.q:
        return finish("untrained", "Màn này chưa có policy. Mở Huấn luyện RL để huấn luyện trên màn hiện tại.")
    if model.layout != layout_key(level):
        return finish("untrained", "Policy không khớp vị trí tường và đích của màn này.")
    env = SokobanPushEnv(level, start=start, max_pushes=max_pushes)
    env.reset()
    for _ in range(max_pushes):
        if won(level, state):
            return finish("solved")
        key = policy_key(level, state)
        if key in visited:
            return finish("policy_failed", "Policy lặp lại trạng thái. Cần huấn luyện thêm; không tự chuyển sang A*.")
        visited.add(key)
        options = env.options()
        if not options:
            return finish("policy_failed", "Policy đi vào trạng thái không còn lượt đẩy hợp lệ.")
        values = model.q.get(key)
        if not values:
            return finish("policy_failed", "Policy chưa học trạng thái này. Hãy huấn luyện từ vị trí hiện tại.")
        chosen = min(options, key=lambda option: (-values.get(option.index, 0.0), option.index))
        _, _, terminated, truncated, info = env.step(chosen.index)
        actions.extend(info["player_actions"])
        pushes += info["pushes"]
        state = env.state
        if terminated:
            return finish("solved" if won(level, state) else "policy_failed", "" if won(level, state) else "Policy đẩy thùng vào thế kẹt.")
        if truncated:
            break
    return finish("policy_failed", "Policy đạt giới hạn số lần đẩy; chưa giải xong màn.")


def reverse_curriculum(level: Level, seed: int, *, rounds: int = 70, depth: int = 10) -> list[tuple[State, int]]:
    """Reversible pulls generate solvable resets, never a teacher action label."""
    rng = random.Random(seed)
    floor = sorted(level.floor - level.goals)
    collected = {}
    for _ in range(rounds):
        state = State(rng.choice(floor), level.goals)
        for pull_depth in range(1, rng.randint(2, depth) + 1):
            paths = walking_paths(level, state)
            pulls = []
            for box in sorted(state.boxes):
                for dr, dc in DIRECTIONS.values():
                    stand = box[0] + dr, box[1] + dc
                    destination = box[0] + 2 * dr, box[1] + 2 * dc
                    if stand in paths and not level.blocked(destination) and destination not in state.boxes:
                        pulls.append((box, stand, destination))
            if not pulls:
                break
            box, new_box, player = rng.choice(pulls)
            state = State(player, frozenset((state.boxes - {box}) | {new_box}))
            reachable = walking_paths(level, state)
            state = State(rng.choice(sorted(reachable)), state.boxes)
            if not won(level, state):
                key = state_key(state)
                if key not in collected or pull_depth < collected[key][1]:
                    collected[key] = state, pull_depth
    return sorted(collected.values(), key=lambda item: (item[1], state_key(item[0])))


def random_baseline(level: Level, starts: list[State], seed: int, max_pushes: int) -> dict:
    rng = random.Random(seed)
    env = SokobanPushEnv(level, max_pushes=max_pushes)
    successes = 0
    for start in starts:
        env.reset(options={"state": start})
        for _ in range(max_pushes):
            options = env.options()
            if not options:
                break
            _, _, terminated, truncated, info = env.step(rng.choice(options).index)
            if info["is_success"]:
                successes += 1
            if terminated or truncated:
                break
    return {"episodes": len(starts), "successes": successes,
            "success_rate": successes / len(starts) if starts else 0.0}


def train_q(level: Level, start: State | None = None, *, episodes: int = 8000, seed: int = 42,
            max_pushes: int = 40, budget: float = 20.0, alpha: float = .3, gamma: float = .98,
            on_progress: Callable | None = None) -> QModel:
    if episodes < 1 or max_pushes < 1 or budget <= 0:
        raise ValueError("Số episode, giới hạn bước và thời gian phải lớn hơn 0.")
    start = start or level.initial
    if won(level, start) or corner_deadlock(level, start):
        raise ValueError("Chọn trạng thái chưa hoàn thành và không kẹt góc để huấn luyện.")
    began = perf_counter()
    rng = random.Random(seed)
    model = QModel(layout_key(level))
    pool = reverse_curriculum(level, seed)
    held_out = [state for i, (state, _) in enumerate(pool) if i % 7 == 0 and state != start][:24]
    held_keys = {state_key(state) for state in held_out}
    curriculum = [(state, depth) for state, depth in pool if state_key(state) not in held_keys and state != start]
    if not curriculum:
        curriculum = [(start, 1)]
    baseline_starts = [start] * 24
    baseline = random_baseline(level, baseline_starts, seed + 99, max_pushes)
    env = SokobanPushEnv(level, start=start, max_pushes=max_pushes, gamma=gamma)
    window_rewards, window_success = [], []
    transitions, root_episodes, root_wins, completed = 0, 0, 0, 0
    for episode in range(1, episodes + 1):
        if perf_counter() - began >= budget:
            break
        progress = (episode - 1) / max(episodes - 1, 1)
        epsilon = .05 + .95 * (1 - progress) ** 1.5
        root = rng.random() < .4
        available = max(1, min(len(curriculum), int(10 + progress * len(curriculum))))
        initial = start if root else rng.choice(curriculum[:available])[0]
        env.reset(options={"state": initial})
        total_reward, success = 0.0, False
        root_episodes += int(root)
        for _ in range(max_pushes):
            options = env.options()
            if not options:
                break
            key = policy_key(level, env.state)
            row = model.q.setdefault(key, {})
            if rng.random() < epsilon:
                chosen = rng.choice(options)
            else:
                highest = max(row.get(item.index, 0.0) for item in options)
                chosen = rng.choice([item for item in options if row.get(item.index, 0.0) == highest])
            _, reward, terminated, truncated, info = env.step(chosen.index)
            future_row = model.q.get(policy_key(level, env.state), {})
            future = 0.0 if terminated else max((future_row.get(item.index, 0.0) for item in env.options()), default=0.0)
            old = row.get(chosen.index, 0.0)
            row[chosen.index] = old + alpha * (reward + gamma * future - old)
            total_reward += reward
            transitions += 1
            success = info["is_success"]
            if terminated or truncated:
                break
        completed = episode
        root_wins += int(root and success)
        window_rewards.append(total_reward)
        window_success.append(int(success))
        if episode % 500 == 0 or episode == episodes:
            evaluation = rollout(level, model, start, max_pushes=max_pushes)
            row = {"episode": episode, "mean_reward": sum(window_rewards) / len(window_rewards),
                   "training_success_rate": sum(window_success) / len(window_success),
                   "epsilon": round(epsilon, 4), "greedy_root_solved": evaluation.solved,
                   "greedy_root_moves": len(evaluation.actions) if evaluation.solved else None}
            model.history.append(row)
            window_rewards.clear(); window_success.clear()
            if on_progress:
                on_progress(episode, episodes, row)
    held_success = sum(rollout(level, model, state, max_pushes=max_pushes).solved for state in held_out)
    root_result = rollout(level, model, start, max_pushes=max_pushes)
    model.training = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "episodes": completed,
        "requested_episodes": episodes, "seed": seed, "alpha": alpha, "gamma": gamma,
        "max_pushes": max_pushes, "elapsed_seconds": round(perf_counter() - began, 3),
        "transitions": transitions, "learned_states": len(model.q), "root_start": start.to_dict(),
        "root_training_episodes": root_episodes, "root_training_successes": root_wins,
        "root_greedy_solved": root_result.solved,
        "root_greedy_moves": len(root_result.actions) if root_result.solved else None,
        "root_greedy_pushes": root_result.pushes if root_result.solved else None,
        "random_policy_baseline": baseline,
        "held_out_reset_states": len(held_out), "held_out_reset_successes": held_success,
        "held_out_note": "Same wall/goal layout; reset positions excluded from curriculum starts, but states or transitions may be visited during training. Not unseen levels.",
        "scope": "single_wall_goal_layout", "curriculum": "reversible_pulls_without_action_labels",
        "state_abstraction": "exact_player_and_sorted_boxes",
        "reward": "-1/push -0.1/walking_step + potential shaping; +50 success; -30 corner/no-action deadlock",
        "search_demonstrations": False, "inference_search_fallback": False,
    }
    return model
