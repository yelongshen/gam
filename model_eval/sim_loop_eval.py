"""
Headless MuJoCo sim launcher for offline evaluation.

This is a thin wrapper around `gear_sonic/scripts/run_sim_loop.py` that releases
the virtual elastic band before the simulator starts.

WHY THIS EXISTS
---------------
`ElasticBand` is created with `enable = True`, which suspends the robot from a
very stiff virtual spring (kp_pos = 10000) anchored at z = 1.0. Interactively an
operator presses '9' in the MuJoCo viewer to release it so the robot drops and
stands on the ground.

A headless evaluation run has no viewer and therefore no way to press '9', so
the robot stays hanging for the whole episode and every physics metric
(torque / tilt / non-fall) becomes meaningless -- the robot literally cannot
fall over.

Rather than editing the shared `unitree_sdk2py_bridge.py`, this launcher
monkey-patches `ElasticBand.__init__` at runtime, purely inside this
evaluation-only entry point. Core repo behaviour is untouched: interactive
users still get the band enabled by default.

Usage (drop-in replacement for run_sim_loop.py):
    .venv_sim/bin/python sim_loop_eval.py --interface lo --no-enable-onscreen

Set GEAR_KEEP_ELASTIC_BAND=1 to keep the band attached (original behaviour).
"""
import os
import sys

import tyro

from gear_sonic.utils.mujoco_sim import unitree_sdk2py_bridge as _bridge


def _release_elastic_band():
    """Patch ElasticBand so it starts released (equivalent to pressing '9')."""
    if os.environ.get("GEAR_KEEP_ELASTIC_BAND", "0") in ("1", "true", "True", "yes"):
        print("[sim_loop_eval] Elastic band KEPT enabled (GEAR_KEEP_ELASTIC_BAND=1)")
        return

    original_init = _bridge.ElasticBand.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.enable = False  # same effect as the '9' key in the viewer

    _bridge.ElasticBand.__init__ = patched_init
    print("[sim_loop_eval] Elastic band RELEASED at startup "
          "(headless equivalent of pressing '9') — robot stands on the ground",
          flush=True)


def main():
    _release_elastic_band()
    # Import after patching so the simulator picks up the patched class.
    from gear_sonic.scripts.run_sim_loop import ArgsConfig
    from gear_sonic.scripts.run_sim_loop import main as sim_main

    config = tyro.cli(ArgsConfig)
    sim_main(config)


if __name__ == "__main__":
    sys.exit(main())
