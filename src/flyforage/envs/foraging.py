from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin, sqrt

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True, slots=True)
class ForagingConfig:
    arena_size: float = 10.0
    max_steps: int = 500
    initial_energy: float = 1.0
    energy_decay: float = 0.0025
    food_energy: float = 0.35
    food_reward: float = 1.0
    movement_speed: float = 0.18
    turn_rate: float = 0.35
    food_count: int = 4
    consume_radius: float = 0.35


class FlyForageEnv(gym.Env[np.ndarray, int]):
    """Minimal 2D foraging environment for E001.

    Observation:
      0 energy in [0, 1]
      1 normalized distance to nearest food in [0, 1]
      2 sin(relative food bearing)
      3 cos(relative food bearing)
      4 normalized x position [-1, 1]
      5 normalized y position [-1, 1]

    Actions:
      0 forward
      1 turn left
      2 turn right
      3 idle
    """

    metadata = {"render_modes": []}

    def __init__(self, config: ForagingConfig | None = None) -> None:
        super().__init__()
        self.config = config or ForagingConfig()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.position = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.energy = self.config.initial_energy
        self.food = np.zeros((0, 2), dtype=np.float32)
        self.steps = 0
        self.food_collected = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        half = self.config.arena_size / 2.0
        self.position = self.np_random.uniform(-half * 0.25, half * 0.25, size=2).astype(np.float32)
        self.heading = float(self.np_random.uniform(-pi, pi))
        self.energy = self.config.initial_energy
        self.steps = 0
        self.food_collected = 0
        self.food = self.np_random.uniform(-half, half, size=(self.config.food_count, 2)).astype(np.float32)
        return self._observation(), self._info()

    def step(self, action: int):
        self.steps += 1
        reward = -0.001

        if action == 0:
            self.position[0] += cos(self.heading) * self.config.movement_speed
            self.position[1] += sin(self.heading) * self.config.movement_speed
            reward -= 0.001
        elif action == 1:
            self.heading = self._wrap_angle(self.heading + self.config.turn_rate)
        elif action == 2:
            self.heading = self._wrap_angle(self.heading - self.config.turn_rate)
        elif action != 3:
            raise ValueError(f"invalid action: {action}")

        half = self.config.arena_size / 2.0
        clipped = np.clip(self.position, -half, half)
        if not np.array_equal(clipped, self.position):
            reward -= 0.05
            self.position = clipped.astype(np.float32)

        self.energy = max(0.0, self.energy - self.config.energy_decay)
        reward += self._consume_food()

        terminated = self.energy <= 0.0 or self.food.shape[0] == 0
        truncated = self.steps >= self.config.max_steps
        return self._observation(), reward, terminated, truncated, self._info()

    def _consume_food(self) -> float:
        if self.food.shape[0] == 0:
            return 0.0
        distances = np.linalg.norm(self.food - self.position, axis=1)
        consumed = np.flatnonzero(distances <= self.config.consume_radius)
        if consumed.size == 0:
            return 0.0

        count = int(consumed.size)
        self.food = np.delete(self.food, consumed, axis=0)
        self.food_collected += count
        self.energy = min(1.0, self.energy + self.config.food_energy * count)
        return self.config.food_reward * count

    def _observation(self) -> np.ndarray:
        half = self.config.arena_size / 2.0
        if self.food.shape[0] == 0:
            distance_norm = 0.0
            relative = 0.0
        else:
            deltas = self.food - self.position
            distances = np.linalg.norm(deltas, axis=1)
            nearest_idx = int(np.argmin(distances))
            dx, dy = deltas[nearest_idx]
            absolute_bearing = atan2(float(dy), float(dx))
            relative = self._wrap_angle(absolute_bearing - self.heading)
            max_distance = sqrt(2.0) * self.config.arena_size
            distance_norm = min(1.0, float(distances[nearest_idx]) / max_distance)

        return np.array(
            [
                self.energy,
                distance_norm,
                sin(relative),
                cos(relative),
                float(self.position[0] / half),
                float(self.position[1] / half),
            ],
            dtype=np.float32,
        )

    def _info(self) -> dict[str, float | int]:
        return {
            "energy": float(self.energy),
            "food_collected": self.food_collected,
            "steps": self.steps,
        }

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + pi) % (2 * pi) - pi
