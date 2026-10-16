from dataclasses import replace
from io import BytesIO
import json
from heapq import heappop, heappush
from itertools import count

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from puzzle.engine import DIRECTIONS, PRESETS, State, from_ascii, step, won
from puzzle.generator import generate
from puzzle.planning import push_actions
from puzzle.q_learning import QModel, layout_key, load_model, rollout, save_model, state_key, train_q
from puzzle.rl_env import SokobanPushEnv
from puzzle.solver import replay, solve

TRADEOFF = "#######\n# #  @#\n# $   #\n#     #\n#. $  #\n# .   #\n#######"


def primitive_dijkstra(level, pushes_first):
    """Independent oracle over individual player moves (no macro or heuristic)."""
    best = {level.initial: (0, 0)}
    serial = count()
    heap = [((0, 0), next(serial), level.initial)]
    while heap:
        cost, _, state = heappop(heap)
        if best[state] != cost:
            continue
        if won(level, state):
            return cost
        for direction in DIRECTIONS:
            movement = step(level, state, direction)
            if not movement.valid:
                continue
            edge = (int(movement.pushed), 1) if pushes_first else (1, int(movement.pushed))
            candidate = cost[0] + edge[0], cost[1] + edge[1]
            if candidate < best.get(movement.state, (float("inf"), float("inf"))):
                best[movement.state] = candidate
                heappush(heap, (candidate, next(serial), movement.state))
    return None


@pytest.mark.parametrize("algorithm,expected", [("astar_moves", (20, 9)), ("astar_pushes", (7, 22))])
def test_astar_objective_matches_independent_step_oracle(algorithm, expected):
    level = from_ascii(TRADEOFF)
    result = solve(level, algorithm=algorithm)
    assert result.solved
    pushes_first = algorithm == "astar_pushes"
    cost = (result.pushes, len(result.actions)) if pushes_first else (len(result.actions), result.pushes)
    assert cost == expected == primitive_dijkstra(level, pushes_first)
    assert won(level, replay(level, result)[-1])


@pytest.mark.parametrize("level", [*PRESETS, generate(60, template="Lối vòng").level])
def test_astar_keeps_bfs_shortest_length_with_fewer_states(level):
    baseline = solve(level)
    optimized = solve(level, algorithm="astar_moves")
    assert optimized.solved and len(optimized.actions) == len(baseline.actions)
    assert optimized.explored <= baseline.explored + 1


def test_astar_limits_and_unsolvable_are_distinct():
    level = from_ascii(TRADEOFF)
    assert solve(level, algorithm="astar_moves", max_states=1).status == "limit_reached"
    assert solve(level, algorithm="astar_moves", timeout=0).status == "limit_reached"
    dead = from_ascii("######\n#$ . #\n# @  #\n#    #\n######")
    assert solve(dead, algorithm="astar_moves").status == "no_solution"


def test_every_macro_is_a_valid_engine_path():
    level = PRESETS[2]
    for macro in push_actions(level, level.initial):
        state, pushes = level.initial, 0
        for action in macro.actions:
            movement = step(level, state, action)
            assert movement.valid
            state = movement.state
            pushes += int(movement.pushed)
        assert state == macro.state and pushes == 1


def test_environment_passes_gymnasium_checker():
    check_env(SokobanPushEnv(PRESETS[1]), skip_render_check=True)


def test_gym_action_mask_reward_and_terminal_transition():
    env = SokobanPushEnv(PRESETS[0])
    observation, info = env.reset(seed=5)
    assert env.observation_space.contains(observation)
    option = next(option for option in env.options() if won(env.level, option.state))
    assert info["action_mask"][option.index] == 1
    observation, reward, terminated, truncated, info = env.step(option.index)
    assert reward > 0 and terminated and not truncated and info["is_success"]
    assert env.observation_space.contains(observation)
    with pytest.raises(RuntimeError):
        env.step(option.index)


def test_time_limit_truncates_instead_of_marking_failure_terminal():
    env = SokobanPushEnv(PRESETS[1], max_pushes=1)
    env.reset()
    nonwinning = next(option for option in env.options() if not won(env.level, option.state))
    _, _, terminated, truncated, _ = env.step(nonwinning.index)
    assert truncated and not terminated


def test_invalid_masked_action_is_penalized_without_moving():
    env = SokobanPushEnv(PRESETS[1])
    _, info = env.reset()
    action = int(np.flatnonzero(info["action_mask"] == 0)[0])
    previous = env.state
    _, reward, _, _, _ = env.step(action)
    assert env.state == previous and reward < 0


def test_q_learning_learns_without_calling_a_search_solver(monkeypatch):
    import puzzle.solver as solver_module
    def forbidden(*args, **kwargs):
        raise AssertionError("RL must not call an oracle solver")
    monkeypatch.setattr(solver_module, "solve", forbidden)
    monkeypatch.setattr(solver_module, "solve_astar", forbidden)
    level = PRESETS[1]
    model = train_q(level, episodes=4000, seed=42, budget=10)
    assert model.q and model.training["transitions"] > 0
    assert model.training["episodes"] == 4000
    assert model.training["search_demonstrations"] is False
    result = rollout(level, model)
    assert result.solved and result.optimal_for is None
    assert won(level, replay(level, result)[-1])
    assert result.to_dict()["algorithm"].startswith("RL")


def test_untrained_or_failed_policy_never_falls_back_to_search():
    level = PRESETS[2]
    assert rollout(level, None).status == "untrained"
    model = QModel(layout_key(level), {state_key(level.initial): {0: 1.0}})
    result = rollout(level, model)
    assert result.status in ("policy_failed", "solved")
    assert result.metadata["inference_uses_search_solver"] is False
    wrong = QModel("wrong-layout", {"something": {0: 1}})
    assert rollout(level, wrong).status == "untrained"


def test_checkpoint_roundtrip_and_layout_scope(tmp_path, monkeypatch):
    import puzzle.q_learning as q_module
    monkeypatch.setattr(q_module, "MODEL_DIR", tmp_path)
    level = PRESETS[0]
    model = train_q(level, episodes=1000, budget=5)
    path = save_model(level, model)
    assert path.exists()
    restored = load_model(level)
    assert restored.q == model.q and restored.history == model.history
    assert rollout(level, restored).solved
    assert load_model(PRESETS[1]) is None
    imported = QModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert imported.training["state_abstraction"] == "exact_player_and_sorted_boxes"


def test_bundled_checkpoint_solves_users_seed_60_without_search():
    level = generate(60, template="Lối vòng").level
    model = load_model(level)
    assert model is not None and model.training["episodes"] == 50000
    result = rollout(level, model)
    assert result.solved and len(result.actions) == 30 and result.pushes == 8
    assert result.metadata["inference_uses_search_solver"] is False


def test_training_is_reproducible_and_logs_real_transitions():
    one = train_q(PRESETS[0], episodes=1000, seed=12, budget=5)
    two = train_q(PRESETS[0], episodes=1000, seed=12, budget=5)
    assert one.q == two.q and one.history == two.history
    assert len(one.history) == 2 and one.history[-1]["episode"] == 1000
    assert one.training["learned_states"] == len(one.q)
    assert one.training["root_training_episodes"] <= one.training["episodes"]
