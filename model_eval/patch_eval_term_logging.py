#!/usr/bin/env python3
"""Add per-clip TERMINATION-REASON logging to the sim evaluation callback (gated by an env var, off by default).

  GEAR_SONIC_LOG_TERMS=/path/to/terms.jsonl  python gear_sonic/eval_agent_trl.py ...

For every env that terminates (first time only) one JSON line is appended:
  {"motion": <clip key>, "step": <eval step>, "terms": {<termination name>: bool, ...},
   "fired": [<names that are True>], "root_z": <robot root height>}

Patches GR00T-WholeBodyControl/gear_sonic/trl/callbacks/im_eval_callback.py in place (a backup
`im_eval_callback.py.bak_termlog` is kept).  Idempotent.
"""
import sys
from pathlib import Path

p = Path("/home/grease/GR00T-WholeBodyControl/gear_sonic/trl/callbacks/im_eval_callback.py")
src = p.read_text()
MARK = "# ===== TERMINATION-REASON LOG (GEAR_SONIC_LOG_TERMS) ====="
if MARK in src:
    print("already patched"); sys.exit(0)
anchor = "        self.terminate_state = torch.logical_or(termination_state, self.terminate_state)\n"
assert src.count(anchor) == 1, "anchor not unique"
block = anchor + f'''
        {MARK}
        import os as _os_tl
        _tl_path = _os_tl.environ.get("GEAR_SONIC_LOG_TERMS")
        if _tl_path:
            import json as _json_tl
            _prev = getattr(self, "_tl_prev_term", None)
            if _prev is None:
                _prev = torch.zeros_like(self.terminate_state, dtype=torch.bool)
            if int(self.curr_steps) <= 6:      # per-step foot tracking error of EVERY env (before any reset happens to it)
                try:
                    from gear_sonic.envs.manager_env.mdp.terminations import _get_body_indexes as _gbi2
                    _cmd2 = self.env.env.command_manager.get_term("motion")
                    _ti2 = _gbi2(_cmd2, ["left_ankle_roll_link", "right_ankle_roll_link"])
                    _e2 = (_cmd2.body_pos_relative_w[:, _ti2] - _cmd2.robot_body_pos_w[:, _ti2]).norm(dim=-1)
                    _dz = (_cmd2.body_pos_relative_w[:, _ti2, 2] - _cmd2.robot_body_pos_w[:, _ti2, 2])
                    with open(_tl_path + ".trace", "a") as _fh2:
                        _fh2.write(_json_tl.dumps({{"step": int(self.curr_steps), "err": _e2.tolist(), "dz": _dz.tolist(),
                                                   "mids": [int(x) for x in self.env.motion_ids.tolist()]}}) + "\\n")
                except Exception as _e:
                    pass
            _new = self.terminate_state.bool() & ~_prev
            if _new.any():
                _tm = self.env.env.termination_manager
                _keys = self.env._motion_lib._motion_data_keys
                _rootz = self.env.env.scene["robot"].data.root_pos_w[:, 2]
                with open(_tl_path, "a") as _fh:
                    for _i in _new.nonzero().flatten().tolist():
                        _terms = {{n: bool(_tm.get_term(n)[_i].item()) for n in _tm.active_terms}}
                        try:
                            from gear_sonic.envs.manager_env.mdp.terminations import _get_body_indexes as _gbi
                            _cmd = self.env.env.command_manager.get_term("motion")
                            _ti = _gbi(_cmd, ["left_ankle_roll_link", "right_ankle_roll_link"])
                            _ferr = (_cmd.body_pos_relative_w[_i, _ti] - _cmd.robot_body_pos_w[_i, _ti]).norm(dim=-1).tolist()
                        except Exception as _e:
                            _ferr = [float("nan"), float("nan")]
                        _fh.write(_json_tl.dumps({{
                            "motion": str(_keys[int(self.env.motion_ids[_i].item())]),
                            "env": int(_i), "mid": int(self.env.motion_ids[_i].item()),
                            "step": int(self.curr_steps), "terms": _terms, "foot_err": _ferr,
                            "fired": [n for n, v in _terms.items() if v and n != "time_out"],
                            "root_z": float(_rootz[_i].item())}}) + "\\n")
            self._tl_prev_term = self.terminate_state.bool().clone()
        # ===== END TERMINATION-REASON LOG =====
'''
p.with_suffix(".py.bak_termlog").write_text(src)
p.write_text(src.replace(anchor, block))
print("patched", p)
