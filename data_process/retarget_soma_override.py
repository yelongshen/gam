#!/usr/bin/env python3
"""Run soma-retargeter's batched BVH->CSV conversion with OVERRIDDEN IK weights.

Why: the LAFAN1 / smpl_filtered pipeline only has joint POSITIONS. Joint ROTATIONS in the SOMA BVH
are reconstructed geometrically, and the twist about each bone (arms, hands, forearms) and the
rest-pose convention (pelvis / chest) are not observable -> garbage orientation targets. The stock
G1 retargeter config weights orientation objectives for Hips/Chest/ForeArm/Hand heavily
(r_weight 2.0 / 0.7 / 1.0 / 1.2), which drives the arms into limit-pinned, twisted configurations
and tilts the pelvis ~25 deg. Position-only objectives for those bodies are well-posed.

Usage (run with soma-retargeter's venv, cwd = ~/soma-retargeter):
  RW='{"Hips":0.0,"Chest":0.0,"LeftForeArm":0.0,"RightForeArm":0.0,"LeftHand":0.0,"RightHand":0.0}' \
  .venv/bin/python ~/gam/data_process/retarget_soma_override.py --config <cfg.json> --viewer null

  RW = {ik_map body: r_weight}   TW = {ik_map body: t_weight}   (both optional JSON)
"""
import json
import os
import sys

APP = os.path.expanduser("~/soma-retargeter/app")
sys.path.insert(0, APP)
sys.path.insert(0, os.path.expanduser("~/soma-retargeter"))

import soma_retargeter.pipelines.utils as pipeline_utils  # noqa: E402

_orig = pipeline_utils.get_retargeter_config
RW = json.loads(os.environ.get("RW", "{}"))
TW = json.loads(os.environ.get("TW", "{}"))


def patched(source, target):
    cfg = _orig(source, target)
    for body, w in RW.items():
        cfg["ik_map"][body]["r_weight"] = float(w)
    for body, w in TW.items():
        cfg["ik_map"][body]["t_weight"] = float(w)
    print(f"[override] r_weight={RW} t_weight={TW}", flush=True)
    return cfg


pipeline_utils.get_retargeter_config = patched

import bvh_to_csv_converter  # noqa: E402

if __name__ == "__main__":
    bvh_to_csv_converter.main()
