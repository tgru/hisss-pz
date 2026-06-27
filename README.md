# hisss-pz

A [PettingZoo](https://pettingzoo.farama.org/) `ParallelEnv` that wraps the
[hisss](https://github.com/ymahlau/hisss) for multi-agent reinforcement learning.

## Install

```bash
uv add hisss-pz
```

## Quickstart

```python
from hisss_pz import battlesnake_duel_v0

env = battlesnake_duel_v0.env()
observations, infos = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```