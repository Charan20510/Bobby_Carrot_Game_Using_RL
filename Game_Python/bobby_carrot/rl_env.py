from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from .game import (
    Bobby,
    FRAMES,
    FRAMES_PER_STEP,
    HEIGHT_POINTS,
    HEIGHT_POINTS_DELTA,
    Map,
    MapInfo,
    State,
    VIEW_HEIGHT,
    VIEW_HEIGHT_POINTS,
    VIEW_WIDTH,
    VIEW_WIDTH_POINTS,
    WIDTH_POINTS,
    WIDTH_POINTS_DELTA,
    load_image,
)


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
    new_best_target_distance_scale: float = 0.4
    new_best_finish_distance_scale: float = 0.8
    post_collection_step_penalty: float = -0.25
    no_progress_penalty_after: int = 40
    no_progress_penalty: float = -0.2
    no_progress_penalty_hard_after: int = 100
    no_progress_penalty_hard: float = -0.4
    all_collected_bonus: float = 15.0


class _EnvRenderAssets:
    def __init__(self) -> None:
        self.bobby_idle = load_image("image/bobby_idle.png")
        self.bobby_death = load_image("image/bobby_death.png")
        self.bobby_fade = load_image("image/bobby_fade.png")
        self.bobby_left = load_image("image/bobby_left.png")
        self.bobby_right = load_image("image/bobby_right.png")
        self.bobby_up = load_image("image/bobby_up.png")
        self.bobby_down = load_image("image/bobby_down.png")
        self.tile_conveyor_left = load_image("image/tile_conveyor_left.png")
        self.tile_conveyor_right = load_image("image/tile_conveyor_right.png")
        self.tile_conveyor_up = load_image("image/tile_conveyor_up.png")
        self.tile_conveyor_down = load_image("image/tile_conveyor_down.png")
        self.tileset = load_image("image/tileset.png")
        self.tile_finish = load_image("image/tile_finish.png")
        self.hud = load_image("image/hud.png")
        self.numbers = load_image("image/numbers.png")


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
        observation_mode: str = "compact",  # "compact", "local" or "full"
        local_view_size: int = 3,
        include_inventory: bool = False,
        headless: bool = True,
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 1000,
        loop_window: int = 12,
        debug: bool = False,
        debug_every: int = 100,
    ) -> None:
        if observation_mode not in {"compact", "local", "full"}:
            raise ValueError("observation_mode must be 'compact', 'local' or 'full'")
        if local_view_size % 2 == 0:
            raise ValueError("local_view_size must be odd")

        self.map_obj = Map(map_kind, map_number)
        self.observation_mode = observation_mode
        self.local_view_size = local_view_size
        self.include_inventory = include_inventory
        self.headless = headless
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps
        self.loop_window = loop_window
        self.debug = debug
        self.debug_every = max(1, debug_every)

        self.frame = 0
        self.step_count = 0
        self.episode_done = False
        self.level_completed = False

        self._map_info_template: Optional[MapInfo] = None
        self.map_info: Optional[MapInfo] = None
        self.bobby: Optional[Bobby] = None
        self.recent_positions: Deque[Tuple[int, int]] = deque(maxlen=self.loop_window)
        self.invalid_streak = 0
        self.steps_since_progress = 0
        self.best_target_distance: Optional[int] = None
        self.best_finish_distance: Optional[int] = None
        self.target_positions: set[Tuple[int, int]] = set()
        self.finish_positions: set[Tuple[int, int]] = set()
        self.cached_targets_tile: Optional[set[int]] = None

        # Rendering is optional and lazily initialized.
        self._pygame = None
        self._screen = None
        self._render_assets: Optional[_EnvRenderAssets] = None
        self._render_start_ticks: Optional[int] = None

    @property
    def action_space_n(self) -> int:
        return 4  # Removed NOOP: only LEFT, RIGHT, UP, DOWN

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
        self.invalid_streak = 0
        self.steps_since_progress = 0
        self.best_target_distance = None
        self.best_finish_distance = None
        self.recent_positions.clear()

        self.bobby = Bobby(start_frame=self.frame, start_time=0, coord_src=self.map_info.coord_start)
        self.bobby.state = State.Down
        self.bobby.coord_dest = self.bobby.coord_src
        self.recent_positions.append(self.bobby.coord_src)
        self._cache_target_positions()
        self._cache_finish_positions()
        self.best_target_distance = self._min_distance_to_target_cached(self.bobby.coord_src)
        self.best_finish_distance = self._min_distance_to_finish(self.bobby.coord_src)

        return self._get_observation()

    def set_map(self, map_kind: str, map_number: int) -> None:
        self.map_obj = Map(map_kind, map_number)

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
        before_pos = self.bobby.coord_src
        was_all_collected = self.bobby.is_finished(self.map_info)
        dist_before = self._phase_distance(before_pos, was_all_collected)

        invalid_move = self._apply_action(action)
        if invalid_move:
            reward += self.reward_config.invalid_move
            info["invalid_move"] = True
        else:
            info["invalid_move"] = False

        self._advance_until_transition()
        after_pos = self.bobby.coord_src
        now_all_collected = self.bobby.is_finished(self.map_info)
        dist_after = self._phase_distance(after_pos, now_all_collected)

        moved = before_pos != after_pos
        if moved and dist_before is not None and dist_after is not None:
            info["distance_delta"] = float(dist_before - dist_after)

        # Grant progress reward only when a genuinely new best distance is reached.
        if not now_all_collected:
            if dist_after is not None:
                if self.best_target_distance is None:
                    self.best_target_distance = dist_after
                elif dist_after < self.best_target_distance:
                    improvement = self.best_target_distance - dist_after
                    reward += self.reward_config.new_best_target_distance_scale * float(improvement)
                    self.best_target_distance = dist_after
        else:
            if dist_after is not None:
                if self.best_finish_distance is None:
                    self.best_finish_distance = dist_after
                elif dist_after < self.best_finish_distance:
                    improvement = self.best_finish_distance - dist_after
                    reward += self.reward_config.new_best_finish_distance_scale * float(improvement)
                    self.best_finish_distance = dist_after

        if now_all_collected and not self._can_start_finish():
            reward += self.reward_config.post_collection_step_penalty
        
        self.recent_positions.append(after_pos)
        info["distance_before"] = dist_before
        info["distance_after"] = dist_after
        info["all_collected"] = now_all_collected

        carrot_delta = self.bobby.carrot_count - before_carrot
        egg_delta = self.bobby.egg_count - before_egg

        if carrot_delta > 0:
            reward += self.reward_config.carrot * carrot_delta
            self._cache_target_positions()
        if egg_delta > 0:
            reward += self.reward_config.egg * egg_delta
            self._cache_target_positions()

        if carrot_delta > 0 or egg_delta > 0:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1

        if now_all_collected and not was_all_collected:
            reward += self.reward_config.all_collected_bonus

        if self.steps_since_progress >= self.reward_config.no_progress_penalty_after:
            reward += self.reward_config.no_progress_penalty
        if self.steps_since_progress >= self.reward_config.no_progress_penalty_hard_after:
            reward += self.reward_config.no_progress_penalty_hard

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
        info["position"] = after_pos
        info["moved"] = moved
        info["steps_since_progress"] = self.steps_since_progress

        if self.debug and (self.step_count % self.debug_every == 0 or info["invalid_move"] or info["dead"]):
            print(f"[env-debug] step={self.step_count} reward={reward:.2f} info={info}")

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
            self._screen = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
            pygame.display.set_caption("BobbyCarrotEnv")
            self._render_assets = _EnvRenderAssets()
            self._render_start_ticks = pygame.time.get_ticks()

        screen = self._screen
        assert screen is not None
        assert self._render_assets is not None

        assets = self._render_assets
        Rect = pygame.Rect

        # Match the main game camera behavior.
        step = self.frame - self.bobby.start_frame
        x0 = self.bobby.coord_src[0] * 32
        y0 = self.bobby.coord_src[1] * 32
        x1 = self.bobby.coord_dest[0] * 32
        y1 = self.bobby.coord_dest[1] * 32

        if self.bobby.state == State.Death:
            cam_x = (x1 - x0) * 6 // 8 + x0 - (VIEW_WIDTH_POINTS // 2) * 32
            cam_y = (y1 - y0) * 6 // 8 + y0 - (VIEW_HEIGHT_POINTS // 2) * 32
        else:
            cam_x = (x1 - x0) * step // (8 * FRAMES_PER_STEP) + x0 - (VIEW_WIDTH_POINTS // 2) * 32
            cam_y = (y1 - y0) * step // (8 * FRAMES_PER_STEP) + y0 - (VIEW_HEIGHT_POINTS // 2) * 32

        cam_x += 16
        cam_y += 16
        if cam_x < 0:
            cam_x = 0
        if cam_x > WIDTH_POINTS_DELTA * 32:
            cam_x = WIDTH_POINTS_DELTA * 32
        if cam_y < 0:
            cam_y = 0
        if cam_y > HEIGHT_POINTS_DELTA * 32:
            cam_y = HEIGHT_POINTS_DELTA * 32

        x_right_offset = WIDTH_POINTS_DELTA * 32 - cam_x
        y_offset = cam_y

        screen.fill((20, 20, 20))

        # Tile rendering with original sprites.
        for x in range(WIDTH_POINTS):
            for y in range(HEIGHT_POINTS):
                item = self.map_info.data[x + y * 16]
                texture = assets.tileset
                if item == 44 and self.bobby.is_finished(self.map_info):
                    texture = assets.tile_finish
                elif item == 40:
                    texture = assets.tile_conveyor_left
                elif item == 41:
                    texture = assets.tile_conveyor_right
                elif item == 42:
                    texture = assets.tile_conveyor_up
                elif item == 43:
                    texture = assets.tile_conveyor_down

                if (item == 44 and self.bobby.is_finished(self.map_info)) or 40 <= item <= 43:
                    src = Rect(32 * ((self.frame // (FRAMES // 10)) % 4), 0, 32, 32)
                else:
                    src = Rect(32 * (item % 8), 32 * (item // 8), 32, 32)

                dest = Rect(x * 32 - cam_x, y * 32 - cam_y, 32, 32)
                screen.blit(texture, dest, src)

        # Bobby sprite.
        bobby_src, bobby_dest = self.bobby.update_texture_position(self.frame, self.map_info.data)
        bobby_tex = {
            State.Idle: assets.bobby_idle,
            State.Death: assets.bobby_death,
            State.FadeIn: assets.bobby_fade,
            State.FadeOut: assets.bobby_fade,
            State.Left: assets.bobby_left,
            State.Right: assets.bobby_right,
            State.Up: assets.bobby_up,
            State.Down: assets.bobby_down,
        }[self.bobby.state]
        bobby_dest = bobby_dest.move(-cam_x, -cam_y)
        screen.blit(bobby_tex, bobby_dest, bobby_src)

        # HUD in the same style as the main game (targets + timer).
        if self.map_info.carrot_total > 0:
            icon_rect = Rect(0, 0, 46, 44)
            num_left = self.map_info.carrot_total - self.bobby.carrot_count
            icon_width = 46
        else:
            icon_rect = Rect(46, 0, 34, 44)
            num_left = self.map_info.egg_total - self.bobby.egg_count
            icon_width = 34

        screen.blit(assets.hud, (32 * 16 - (icon_width + 4) - x_right_offset, 4 + y_offset), icon_rect)
        num_10 = max(0, min(99, num_left)) // 10
        num_01 = max(0, min(99, num_left)) % 10
        screen.blit(
            assets.numbers,
            (32 * 16 - (icon_width + 4) - 2 - 12 - x_right_offset, 4 + 14 + y_offset),
            Rect(num_01 * 12, 0, 12, 18),
        )
        screen.blit(
            assets.numbers,
            (32 * 16 - (icon_width + 4) - 2 - 12 * 2 - 1 - x_right_offset, 4 + 14 + y_offset),
            Rect(num_10 * 12, 0, 12, 18),
        )

        now_ticks = pygame.time.get_ticks()
        base_ticks = self._render_start_ticks if self._render_start_ticks is not None else now_ticks
        elapsed_secs = max(0, (now_ticks - base_ticks) // 1000)
        elapsed_secs = min(99 * 60 + 99, elapsed_secs)
        minutes = elapsed_secs // 60
        seconds = elapsed_secs % 60
        for idx, offset in enumerate([minutes // 10, minutes % 10, 10, seconds // 10, seconds % 10]):
            screen.blit(assets.numbers, (4 + 12 * idx + cam_x, 4 + y_offset), Rect(offset * 12, 0, 12, 18))

        pygame.display.flip()
        pygame.event.pump()

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None
            self._render_assets = None
            self._render_start_ticks = None

    def _apply_action(self, action: int) -> bool:
        assert self.bobby is not None
        assert self.map_info is not None

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

            if self._can_start_finish() and self.bobby.state != State.FadeOut and not self.bobby.faded_out:
                self.bobby.start_frame = self.frame
                self.bobby.state = State.FadeOut

            if self.bobby.faded_out:
                return

            now_walking = self.bobby.is_walking()
            if was_walking and not now_walking:
                return
            was_walking = now_walking

    def _is_level_completed(self) -> bool:
        # For RL, reaching finish tile after collecting all targets is completion.
        return self._can_start_finish()

    def _can_start_finish(self) -> bool:
        assert self.bobby is not None
        assert self.map_info is not None

        pos = self.bobby.coord_src[0] + self.bobby.coord_src[1] * 16
        on_finish_tile = self.map_info.data[pos] == 44
        return self.bobby.is_finished(self.map_info) and on_finish_tile

    def _get_observation(self) -> np.ndarray:
        assert self.bobby is not None
        assert self.map_info is not None

        px, py = self.bobby.coord_src
        base = [px, py]

        if self.observation_mode == "full":
            tiles = np.array(self.map_info.data, dtype=np.int16)
            if self.include_inventory:
                inv = self._compressed_inventory()
                return np.concatenate([np.array(base + inv, dtype=np.int16), tiles])
            return np.concatenate([np.array(base, dtype=np.int16), tiles])

        half = self.local_view_size // 2
        local = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                xx = px + dx
                yy = py + dy
                if 0 <= xx < 16 and 0 <= yy < 16:
                    tile_val = self.map_info.data[xx + yy * 16]
                    if self.observation_mode == "compact":
                        local.append(self._tile_bucket(tile_val))
                    else:
                        local.append(tile_val)
                else:
                    local.append(-1)

        if self.include_inventory:
            base.extend(self._compressed_inventory())
        return np.concatenate([np.array(base, dtype=np.int16), np.array(local, dtype=np.int16)])

    def _compressed_inventory(self) -> list[int]:
        assert self.bobby is not None
        assert self.map_info is not None

        remaining_carrots = self.map_info.carrot_total - self.bobby.carrot_count
        remaining_eggs = self.map_info.egg_total - self.bobby.egg_count
        remaining_bucket = min(5, max(0, (remaining_carrots + remaining_eggs) // 2))
        return [
            int(self.bobby.key_gray > 0),
            int(self.bobby.key_yellow > 0),
            int(self.bobby.key_red > 0),
            remaining_bucket,
        ]

    def observation_to_key(self, obs: np.ndarray) -> Tuple[int, ...]:
        return tuple(int(x) for x in obs.tolist())

    def _cache_target_positions(self) -> None:
        assert self.map_info is not None
        self.target_positions.clear()
        for y in range(16):
            for x in range(16):
                val = self.map_info.data[x + y * 16]
                if val == 19 and (self.map_info.carrot_total - self.bobby.carrot_count) > 0:
                    self.target_positions.add((x, y))
                elif val == 45 and (self.map_info.egg_total - self.bobby.egg_count) > 0:
                    self.target_positions.add((x, y))

    def _cache_finish_positions(self) -> None:
        assert self.map_info is not None
        self.finish_positions.clear()
        for y in range(16):
            for x in range(16):
                if self.map_info.data[x + y * 16] == 44:
                    self.finish_positions.add((x, y))

    def _min_distance_to_target_cached(self, pos: Tuple[int, int]) -> Optional[int]:
        if not self.target_positions:
            return None
        px, py = pos
        best: Optional[int] = None
        for tx, ty in self.target_positions:
            d = abs(px - tx) + abs(py - ty)
            if best is None or d < best:
                best = d
        return best

    def _min_distance_to_finish(self, pos: Tuple[int, int]) -> Optional[int]:
        if not self.finish_positions:
            return None
        px, py = pos
        best: Optional[int] = None
        for tx, ty in self.finish_positions:
            d = abs(px - tx) + abs(py - ty)
            if best is None or d < best:
                best = d
        return best

    def _phase_distance(self, pos: Tuple[int, int], all_collected: bool) -> Optional[int]:
        if all_collected:
            return self._min_distance_to_finish(pos)
        return self._min_distance_to_target_cached(pos)

    @staticmethod
    def _tile_bucket(tile: int) -> int:
        if tile == -1:
            return -1
        if tile < 18:
            return 0
        if tile in {18, 20, 21, 23, 30, 39}:
            return 1
        if tile == 19:
            return 2
        if tile == 45:
            return 3
        if tile == 44:
            return 4
        if tile in {31, 46}:
            return 5
        if tile in {32, 34, 36}:
            return 6
        if tile in {33, 35, 37}:
            return 7
        if tile in {22, 38}:
            return 8
        if tile in {24, 25, 26, 27, 28, 29, 40, 41, 42, 43}:
            return 9
        return 10

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
