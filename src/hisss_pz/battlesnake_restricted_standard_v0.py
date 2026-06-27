import hisss

from hisss_pz.core import HisssEnv, RenderMode

def env(render_mode: RenderMode | None = None) -> HisssEnv:
    cfg = hisss.restricted_standard_config()

    return HisssEnv("battlesnake_restricted_standard_v0", cfg, render_mode)