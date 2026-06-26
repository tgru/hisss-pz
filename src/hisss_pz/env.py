from __future__ import annotations

from typing import Any, Literal, override

RenderMode = Literal["human", "ansi"]

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

import hisss
from hisss import UP


class HisssEnv(ParallelEnv[str, np.ndarray, int]):
    """PettingZoo Parallel environment wrapping the hisss BattleSnake simulator.

    All agents act simultaneously each step. Dead agents are terminated and
    removed from ``self.agents``; the episode ends when the game is terminal.

    Actions (per agent): 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    Observations: float32 array of shape (width, height, channels), centred on
        the agent's own head.
    """

    render_modes: list[RenderMode] = ["human", "ansi"]
    metadata = {
        "render_modes": render_modes,
        "name": "BattleSnake-Duell-v0",
    }

    def __init__(
        self,
        cfg: hisss.BattleSnakeConfig | None = None,
        render_mode: RenderMode | None = None,
    ) -> None:
        if cfg is None:
            cfg = hisss.duel_config()
        if render_mode is not None and render_mode not in self.render_modes:
            raise ValueError(
                f"render_mode {render_mode!r} is not supported. Valid modes: {self.render_modes}"
            )
        self.cfg = cfg
        self.render_mode: RenderMode | None = render_mode
        self._game: hisss.BattleSnakeGame | None = None

        tmp = hisss.BattleSnakeGame(cfg)
        try:
            obs_arr, _, _ = tmp.get_obs()
        finally:
            tmp.close()

        self._num_snakes: int = obs_arr.shape[0]
        self._obs_shape: tuple[int, ...] = tuple(obs_arr.shape[1:])

        self.possible_agents: list[str] = [
            f"snake_{i}" for i in range(self._num_snakes)
        ]
        self.agents: list[str] = []

    @override
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self._game is not None:
            self._game.close()
        self._game = hisss.BattleSnakeGame(self.cfg)
        self.agents = list(self.possible_agents)

        obs_arr, _, _ = self._game.get_obs()
        observations = {
            f"snake_{i}": obs_arr[i].astype(np.float32)
            for i in range(self._num_snakes)
        }
        infos = {agent: self._agent_info(int(agent.split("_")[1])) for agent in self.agents}
        return observations, infos

    @override
    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        if self._game is None:
            raise RuntimeError("Call reset() before step().")

        alive_before = self._game.players_alive()

        clamped = []
        for i in alive_before:
            a = int(actions.get(f"snake_{i}", UP))
            legal = self._game.available_actions(i)
            clamped.append(a if a in legal else legal[0])
        action_tuple = tuple(clamped)
        rewards_arr, done, _ = self._game.step(action_tuple)

        alive_after = set(self._game.players_alive())
        if not done:
            obs_arr, _, _ = self._game.get_obs()
            at_turn = self._game.players_at_turn()
            obs_by_player = {at_turn[j]: obs_arr[j].astype(np.float32) for j in range(len(at_turn))}
        else:
            obs_by_player = {}
        zero_obs = np.zeros(self._obs_shape, dtype=np.float32)

        observations: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, Any] = {}

        for agent in self.agents:
            idx = int(agent.split("_")[1])
            observations[agent] = obs_by_player.get(idx, zero_obs)
            rewards[agent] = float(rewards_arr[idx])
            terminations[agent] = bool(done or idx not in alive_after)
            truncations[agent] = False
            infos[agent] = self._agent_info(idx)

        self.agents = [] if done else [f"snake_{i}" for i in sorted(alive_after)]

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    @override
    def render(self) -> str | None:
        if self._game is None:
            return None
        if self.render_mode == "human":
            self._game.render()
            return None
        if self.render_mode == "ansi":
            return self._game.get_str_repr()
        return None

    @override
    def close(self) -> None:
        if self._game is not None:
            self._game.close()
            self._game = None

    @override
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._obs_shape,
            dtype=np.float32,
        )

    @override
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(4)

    def _agent_info(self, idx: int) -> dict[str, Any]:
        if self._game is None:
            return {}
        alive = set(self._game.players_alive())
        if idx not in alive:
            return {"alive": False, "health": 0, "length": 0, "turn": 0}
        return {
            "alive": True,
            "health": int(self._game.player_healths()[idx]),
            "length": int(self._game.player_lengths()[idx]),
            "turn": self._game.turns_played,
        }

    def __del__(self) -> None:
        self.close()
