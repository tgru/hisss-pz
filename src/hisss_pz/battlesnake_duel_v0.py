import hisss

from hisss_pz.core import HisssEnv, RenderMode

def env(render_mode: RenderMode | None = None) -> HisssEnv:
    cfg = hisss.duel_config()

    return HisssEnv("battlesnake_duel_v0", cfg, render_mode)