from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .game import Bobby, FRAMES_PER_STEP, Map, MapInfo, State


ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_UP = 2
ACTION_DOWN = 3
ACTION_NOOP = 4

ACTION_TO_STATE = {
    ACTION_LEFT: State.Left,
    ACTION_RIGHT: State.Right,
    ACTION_UP: State.Up,
    ACTION_DOWN: State.Down,
}


@dataclass
class RewardConfig:
    carrot: float = 10.0
    egg: float = 20.0
    finish: float = 50.0
    death: float = -50.0
    step: float = -0.1
    invalid_move: float = -0.5


class BobbyCarrotEnv:
    """Gym-style environment wrapper for the Bobby Carrot game logic.

    Notes:
    - This class reuses existing game logic (Bobby and map tile updates).
    - Training can run in headless mode (default) with no rendering.
    """

    def __init__(
        self,
        map_kind: str = "normal",
        map_number: int = 1,
        observation_mode: str = "local",  # "local" or "full"
        local_view_size: int = 5,
        headless: bool = True,
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 1000,
    ) -> None:
        if observation_mode not in {"local", "full"}:
            raise ValueError("observation_mode must be 'local' or 'full'")
        if local_view_size % 2 == 0:
            raise ValueError("local_view_size must be odd")

        self.map_obj = Map(map_kind, map_number)
        self.observation_mode = observation_mode
        self.local_view_size = local_view_size
        self.headless = headless
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps

        self.frame = 0
        self.step_count = 0
        self.episode_done = False
        self.level_completed = False

        self._map_info_template: Optional[MapInfo] = None
        self.map_info: Optional[MapInfo] = None
        self.bobby: Optional[Bobby] = None

        # Rendering is optional and lazily initialized.
        self._pygame = None
        self._screen = None

    @property
    def action_space_n(self) -> int:
        return 5

    def reset(self) -> np.ndarray:
        fresh = self.map_obj.load_map_info()
        self._map_info_template = fresh
        self.map_info = MapInfo(
            data=fresh.data.copy(),
            coord_start=fresh.coord_start,
            carrot_total=fresh.carrot_total,
            egg_total=fresh.egg_total,
        )

        self.frame = 0
        self.step_count = 0
        self.episode_done = False
        self.level_completed = False

        self.bobby = Bobby(start_frame=self.frame, start_time=0, coord_src=self.map_info.coord_start)
        self.bobby.state = State.Down
        self.bobby.coord_dest = self.bobby.coord_src

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        if self.map_info is None or self.bobby is None:
            raise RuntimeError("Call reset() before step().")
        if self.episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if action < 0 or action >= self.action_space_n:
            raise ValueError(f"Invalid action {action}. Expected [0..{self.action_space_n - 1}].")

        reward = self.reward_config.step
        info: Dict[str, object] = {
            "invalid_move": False,
            "collected_carrot": 0,
            "collected_egg": 0,
            "level_completed": False,
            "dead": False,
        }

        before_carrot = self.bobby.carrot_count
        before_egg = self.bobby.egg_count

        invalid_move = self._apply_action(action)
        if invalid_move:
            reward += self.reward_config.invalid_move
            info["invalid_move"] = True

        self._advance_until_transition()

        carrot_delta = self.bobby.carrot_count - before_carrot
        egg_delta = self.bobby.egg_count - before_egg

        if carrot_delta > 0:
            reward += self.reward_config.carrot * carrot_delta
        if egg_delta > 0:
            reward += self.reward_config.egg * egg_delta

        self.step_count += 1
        done = False

        if self.bobby.dead:
            reward += self.reward_config.death
            done = True
            info["dead"] = True

        if self._is_level_completed():
            reward += self.reward_config.finish
            done = True
            self.level_completed = True
            info["level_completed"] = True

        if self.step_count >= self.max_steps:
            done = True

        self.episode_done = done
        info["collected_carrot"] = carrot_delta
        info["collected_egg"] = egg_delta
        info["steps"] = self.step_count

        return self._get_observation(), float(reward), done, info

    def render(self) -> None:
        """Optional rendering hook.

        For training, keep headless=True and skip rendering.
        """
        if self.headless:
            return
        if self.map_info is None or self.bobby is None:
            return

        import pygame

        if self._pygame is None:
            self._pygame = pygame
            pygame.init()
            self._screen = pygame.display.set_mode((16 * 24, 16 * 24))
            pygame.display.set_caption("BobbyCarrotEnv")

        screen = self._screen
        assert screen is not None

        tile_size = 24
        screen.fill((20, 20, 20))

        for y in range(16):
            for x in range(16):
                val = self.map_info.data[x + y * 16]
                color = self._tile_color(val)
                pygame.draw.rect(
                    screen,
                    color,
                    (x * tile_size, y * tile_size, tile_size - 1, tile_size - 1),
                )

        px, py = self.bobby.coord_src
        pygame.draw.circle(
            screen,
            (255, 140, 0),
            (px * tile_size + tile_size // 2, py * tile_size + tile_size // 2),
            tile_size // 3,
        )
        pygame.display.flip()

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None

    def _apply_action(self, action: int) -> bool:
        assert self.bobby is not None
        assert self.map_info is not None

        if action == ACTION_NOOP:
            return False

        state = ACTION_TO_STATE[action]

        if self.bobby.is_walking():
            self.bobby.update_next_state(state, self.frame)
            return False

        old_dest = self.bobby.coord_dest
        self.bobby.update_state(state, self.frame, self.map_info.data)

        return self.bobby.coord_dest == old_dest

    def _advance_until_transition(self) -> None:
        """Advance internal frames until movement/transition settles.

        We rely on existing Bobby.update_texture_position logic for all tile updates.
        """
        assert self.bobby is not None
        assert self.map_info is not None

        was_walking = self.bobby.is_walking()
        max_internal_frames = 8 * FRAMES_PER_STEP * 3

        for _ in range(max_internal_frames):
            self.frame += 1
            self.bobby.update_texture_position(self.frame, self.map_info.data)

            if self.bobby.dead:
                return

            if self._is_level_completed() and self.bobby.state != State.FadeOut:
                self.bobby.start_frame = self.frame
                self.bobby.state = State.FadeOut

            if self.bobby.faded_out:
                return

            now_walking = self.bobby.is_walking()
            if was_walking and not now_walking:
                return
            was_walking = now_walking

    def _is_level_completed(self) -> bool:
        assert self.bobby is not None
        assert self.map_info is not None

        pos = self.bobby.coord_src[0] + self.bobby.coord_src[1] * 16
        on_finish_tile = self.map_info.data[pos] == 44
        return self.bobby.is_finished(self.map_info) and on_finish_tile and (
            self.bobby.faded_out or self.bobby.state == State.FadeOut
        )

    def _get_observation(self) -> np.ndarray:
        assert self.bobby is not None
        assert self.map_info is not None

        px, py = self.bobby.coord_src
        remaining_carrots = self.map_info.carrot_total - self.bobby.carrot_count
        remaining_eggs = self.map_info.egg_total - self.bobby.egg_count

        inventory = np.array(
            [
                px,
                py,
                self.bobby.carrot_count,
                self.bobby.egg_count,
                self.bobby.key_gray,
                self.bobby.key_yellow,
                self.bobby.key_red,
                remaining_carrots,
                remaining_eggs,
            ],
            dtype=np.int16,
        )

        if self.observation_mode == "full":
            tiles = np.array(self.map_info.data, dtype=np.int16)
            return np.concatenate([inventory, tiles])

        half = self.local_view_size // 2
        local = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                xx = px + dx
                yy = py + dy
                if 0 <= xx < 16 and 0 <= yy < 16:
                    local.append(self.map_info.data[xx + yy * 16])
                else:
                    local.append(-1)

        return np.concatenate([inventory, np.array(local, dtype=np.int16)])

    @staticmethod
    def _tile_color(tile: int) -> Tuple[int, int, int]:
        if tile < 18:
            return (40, 40, 40)
        if tile == 19:
            return (255, 180, 0)
        if tile == 45:
            return (250, 250, 250)
        if tile == 44:
            return (0, 180, 255)
        if tile in {31, 46}:
            return (200, 30, 30)
        if tile in {32, 34, 36}:
            return (30, 200, 30)
        if tile in {33, 35, 37}:
            return (130, 100, 30)
        return (110, 110, 110)
