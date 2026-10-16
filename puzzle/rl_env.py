"""Gymnasium Sokoban environment with masked push-level actions."""
from __future__ import annotations

from itertools import permutations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .engine import Level, State, corner_deadlock, to_ascii, won
from .planning import PushAction, push_actions


class SokobanPushEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, level: Level, *, start: State | None = None, max_pushes: int = 40,
                 gamma: float = .98, render_mode: str | None = None):
        super().__init__()
        self.level, self.start = level, start or level.initial
        self.state = self.start
        self.max_pushes, self.gamma, self.render_mode = max_pushes, gamma, render_mode
        self.action_space = spaces.Discrete(len(level.goals) * 4)
        self.observation_space = spaces.MultiBinary((4, level.rows, level.cols))
        self._options: dict[State, tuple[PushAction, ...]] = {}
        self._potentials: dict[frozenset, float] = {}
        self.steps = 0
        self._finished = False

    def options(self, state: State | None = None) -> tuple[PushAction, ...]:
        state = state or self.state
        if state not in self._options:
            self._options[state] = push_actions(self.level, state)
        return self._options[state]

    def potential(self, state: State) -> float:
        if state.boxes not in self._potentials:
            goals = sorted(self.level.goals)
            self._potentials[state.boxes] = -float(min(
                sum(abs(box[0] - goal[0]) + abs(box[1] - goal[1]) for box, goal in zip(sorted(state.boxes), order))
                for order in permutations(goals)))
        return self._potentials[state.boxes]

    def _observation(self):
        observation = np.zeros(self.observation_space.shape, dtype=np.int8)
        for channel, cells in enumerate((self.level.walls, self.level.goals, self.state.boxes, {self.state.player})):
            for row, col in cells:
                observation[channel, row, col] = 1
        return observation

    def _info(self):
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for option in self.options():
            mask[option.index] = 1
        return {"action_mask": mask, "is_success": won(self.level, self.state)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = (options or {}).get("state", self.start)
        self.steps = 0
        self._finished = False
        return self._observation(), self._info()

    def step(self, action):
        if self._finished:
            raise RuntimeError("Episode đã kết thúc; cần gọi reset().")
        if not self.action_space.contains(action):
            raise ValueError("Action ngoài không gian hành động.")
        self.steps += 1
        option = next((item for item in self.options() if item.index == int(action)), None)
        actions = ()
        if option is None:
            reward = -2.0
            terminated = won(self.level, self.state)
        else:
            previous = self.state
            self.state = option.state
            actions = option.actions
            success = won(self.level, self.state)
            dead = corner_deadlock(self.level, self.state) or (not success and not self.options())
            terminated = success or dead
            next_potential = 0.0 if terminated else self.potential(self.state)
            shaping = .5 * (self.gamma * next_potential - self.potential(previous))
            reward = -1.0 - .1 * (len(actions) - 1) + shaping
            if success:
                reward += 50.0
            elif dead:
                reward -= 30.0
        truncated = self.steps >= self.max_pushes and not terminated
        self._finished = terminated or truncated
        info = self._info()
        info.update({"player_actions": actions, "pushes": int(option is not None)})
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "ansi":
            return to_ascii(self.level, self.state)
        if self.render_mode == "rgb_array":
            from .rendering import render_board
            return np.asarray(render_board(self.level, self.state))
        return None
