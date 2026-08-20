"""
Offline simulation evaluation orchestrator for the split_test dataset.

Drives the full MuJoCo-in-the-loop pipeline per clip:
  1. (auto) MuJoCo sim  -> run_sim_loop.py (LowState over DDS on 'lo')
  2. (per clip) policy  -> g1_deploy_onnx_ref (Mode-2 zmq, CSV logs)
        - inject ']' on stdin to enter CONTROL state (activation gate)
  3. (per clip) publisher -> stream_clip_mode2.py (FK smpl_joints -> ZMQ)
Parses per-clip CSV logs (sim_eval_metrics) and aggregates per category.

Resumable: appends each clip result to --out_csv immediately and skips clips
already present on restart.

Usage:
  .venv_sim/bin/python run_sim_eval.py --per_category 400
  .venv_sim/bin/python run_sim_eval.py --per_category 0   # ALL clips
  .venv_sim/bin/python run_sim_eval.py --aggregate_only   # just rebuild report
"""
import os
import csv
import sys
import time
import signal
import argparse
import subprocess
from collections import defaultdict

import numpy as np
import sim_eval_metrics as M

REPO = "/home/grease/gam"
DEPLOY = os.path.join(REPO, "gear_sonic_deploy")
BIN = "./target/release/g1_deploy_onnx_ref"
SIM_PY = os.path.join(REPO, ".venv_sim/bin/python")
TELEOP_PY = os.path.join(REPO, ".venv_teleop/bin/python")
POLICY_ARGS = [
    "lo", "policy/low_latency/model_decoder.onnx", "reference/example/",
    "--obs-config", "policy/low_latency/observation_config.yaml",
    "--encoder-file", "policy/low_latency/model_encoder.onnx",
    "--input-type", "zmq", "--zmq-host", "localhost", "--zmq-port", "5556",
    "--zmq-topic", "pose", "--zmq-conflate", "--disable-crc-check",
    "--enable-csv-logs",
]
RESULT_KEYS = ['category', 'clip', 'n_frames', 'duration_s', 'track_mae_deg',
               'root_angvel_mean', 'max_tilt_deg', 'final_tilt_deg', 'non_fall',
               'action_jerk', 'mean_torque_Nm', 'peak_torque_Nm',
               'saw_mode2', 'saw_playing', 'tracked_frac',
               'start_tilt_deg', 'settle_max_tilt_deg', 'clean_start', 'valid']


def abspath(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO, p))


def wait_for(logpath, needle, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with open(logpath, 'rb') as f:
                if needle.encode() in f.read():
                    return True
        except FileNotFoundError:
            pass
        time.sleep(0.5)
    return False


def sim_running():
    """True if the MuJoCo sim process is alive.

    NOTE: `pgrep -f <pat>` run via a shell also matches the shell's OWN command
    line (which contains <pat>), so a naive check always returns True. The
    bracket trick ("[s]im_loop_eval") makes the literal text in the shell's
    argv not match the regex, so only the real sim process is found.
    """
    return os.system(
        "pgrep -f '[s]im_loop_eval\\.py|[r]un_sim_loop\\.py' >/dev/null") == 0


def ensure_sim(timeout=120, settle=45):
    """Ensure the MuJoCo sim is running. Started fully detached (new session)
    so it is never killed by signals sent to the policy/orchestrator group."""
    if sim_running():
        return True
    print("[sim] starting MuJoCo sim (headless, detached)...", flush=True)
    env = dict(os.environ)
    env.pop("CYCLONEDDS_HOME", None)
    env["MUJOCO_GL"] = "egl"
    lf = open("/tmp/sim_loop.log", "wb")
    # sim_loop_eval.py is a thin wrapper around run_sim_loop.py that releases
    # the virtual elastic band (headless equivalent of pressing '9'). Without
    # it the robot hangs from a stiff spring and can never fall, making every
    # physics metric meaningless.
    subprocess.Popen([SIM_PY, "model_eval/sim_loop_eval.py",
                      "--interface", "lo", "--no-enable-onscreen"],
                     cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, env=env,
                     start_new_session=True)   # <-- detach (setsid)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if sim_running():
            time.sleep(settle)  # let MuJoCo finish loading + start publishing
            if sim_running():
                return True
            # died during startup -> try once more
            print("[sim] died during startup, retrying...", flush=True)
            return ensure_sim(timeout=timeout, settle=settle)
        time.sleep(2)
    return False


def restart_sim():
    """Kill and relaunch the simulator to reset the robot to a standing pose.

    The sim keeps its state across policy runs, so once the robot falls it
    stays on the floor and every following clip starts from a fallen robot.
    Restarting is the only reliable way to reset it headlessly.
    """
    os.system("pkill -9 -f 'sim_loop_eval|run_sim_loop' >/dev/null 2>&1")
    time.sleep(6)   # let DDS sockets fully release before rebinding
    return ensure_sim()


def run_clip(clip_path, tag, fps=50.0, settle=1.0, max_wait=180):
    logs_dir = f"logs/{tag}"
    abs_logs = os.path.join(DEPLOY, logs_dir)
    os.system(f"rm -rf {abs_logs}")
    plog = f"/tmp/policy_{tag}.log"
    env = dict(os.environ); env.pop("CYCLONEDDS_HOME", None)
    lf = open(plog, 'wb')
    pol = subprocess.Popen([BIN] + POLICY_ARGS + ["--logs-dir", logs_dir],
                           cwd=DEPLOY, stdin=subprocess.PIPE, stdout=lf,
                           stderr=subprocess.STDOUT, env=env)
    try:
        if not wait_for(plog, "Init Done", timeout=90):
            pol.send_signal(signal.SIGINT)
            try: pol.wait(timeout=10)
            except Exception: pol.kill()
            return None
        # Activation handshake:
        #   ']'  -> operator_state.start = true  (INIT/WAIT_FOR_CONTROL -> CONTROL)
        #   '\n' -> toggle_zmq_mode = true       (use_zmq_stream = true)
        # WITHOUT the newline the policy ignores the ZMQ stream entirely and
        # keeps tracking the default reference motion (encoder_mode stays 0).
        pol.stdin.write(b']'); pol.stdin.flush()
        time.sleep(1.5)
        pol.stdin.write(b'\n'); pol.stdin.flush()
        time.sleep(1.5)
        # verify the toggle actually took effect before streaming
        if not wait_for(plog, "ZMQ STREAMING MODE: ENABLED", timeout=10):
            print("   [!] ZMQ streaming toggle did not engage — retrying newline",
                  flush=True)
            pol.stdin.write(b'\n'); pol.stdin.flush()
            time.sleep(2.0)
        env2 = dict(os.environ); env2.pop("CYCLONEDDS_HOME", None)
        try:
            subprocess.run([TELEOP_PY, "data_process/stream_clip_mode2.py", "--path", clip_path,
                            "--fps", str(fps), "--settle", str(settle)],
                           cwd=REPO, env=env2, timeout=max_wait,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.5)
        try:
            pol.stdin.write(b'o'); pol.stdin.flush()
        except Exception:
            pass
        time.sleep(0.3)
    finally:
        pol.send_signal(signal.SIGINT)
        try: pol.wait(timeout=15)
        except subprocess.TimeoutExpired: pol.kill()
        lf.close()
    # clean the big per-clip logs to save disk once metrics are extracted
    return abs_logs


def load_done(out_csv):
    done = set()
    rows = []
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                done.add((r['category'], r['clip']))
                rows.append(r)
    return done, rows


def aggregate(rows, per_category, report_path):
    # Only aggregate clips where the policy actually tracked the streamed
    # motion (encoder mode 2 + motion playing). Anything else means the robot
    # was idling / tracking the built-in reference motion.
    def is_valid(r):
        return str(r.get('valid', '')) in ('1', '1.0', 'True')
    valid_rows = [r for r in rows if is_valid(r)]
    n_invalid = len(rows) - len(valid_rows)

    agg = defaultdict(list)
    for m in valid_rows:
        agg[m['category']].append(m)

    def fget(x, k):
        try:
            v = float(x[k]); return v
        except Exception:
            return float('nan')

    L = []
    L.append("=" * 80)
    L.append("OFFLINE SIMULATION EVALUATION — per-category metrics")
    L.append("=" * 80)
    L.append(f"Clips evaluated: {len(valid_rows)} valid"
             + (f"  ({n_invalid} EXCLUDED as invalid: robot not tracking stream)"
                if n_invalid else "")
             + f"  (target {per_category or 'ALL'}/category)")
    L.append("")
    hdr = f"{'category':24s} {'N':>4s} {'non_fall%':>9s} {'MAE°':>7s} " \
          f"{'tilt°':>6s} {'|w|':>5s} {'torqueNm':>9s} {'peakNm':>7s} {'trk%':>5s}"
    L.append(hdr); L.append("-" * len(hdr))
    for c in sorted(agg.keys()):
        ms = agg[c]
        def mean(k):
            vals = [fget(x, k) for x in ms]
            vals = [v for v in vals if not np.isnan(v)]
            return float(np.mean(vals)) if vals else float('nan')
        L.append(f"{c:24s} {len(ms):4d} {100*mean('non_fall'):8.1f}% "
                 f"{mean('track_mae_deg'):7.1f} {mean('max_tilt_deg'):6.1f} "
                 f"{mean('root_angvel_mean'):5.2f} {mean('mean_torque_Nm'):9.2f} "
                 f"{mean('peak_torque_Nm'):7.1f} {100*mean('tracked_frac'):4.0f}%")
    L.append("=" * 80)
    L.append("Legend: non_fall%=survived, MAE°=action-vs-achieved pose, tilt°=max base tilt,")
    L.append("        |w|=mean root ang vel (rad/s), torque in N·m.")
    L.append("=" * 80)
    rep = "\n".join(L)
    with open(report_path, 'w') as f:
        f.write(rep + "\n")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_csv', default='data_analysis/split/split_test.csv')
    ap.add_argument('--per_category', type=int, default=400)
    ap.add_argument('--out_csv', default='data_analysis/sim_eval/sim_eval_results.csv')
    ap.add_argument('--report', default='data_analysis/sim_eval/sim_eval_report.txt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--aggregate_only', action='store_true')
    args = ap.parse_args()

    done, existing = load_done(args.out_csv)

    if args.aggregate_only:
        rep = aggregate(existing, args.per_category, args.report)
        print(rep); return

    rows = list(csv.DictReader(open(args.split_csv)))
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r['category']].append(r)
    import random
    random.seed(args.seed)
    selected = []
    for c, items in by_cat.items():
        random.shuffle(items)
        take = items if args.per_category == 0 else items[:args.per_category]
        selected.extend(take)

    todo = [r for r in selected
            if (r['category'], os.path.basename(r['path'])) not in done]
    print(f"Total target: {len(selected)} | already done: {len(done)} | "
          f"remaining: {len(todo)}")

    if not ensure_sim():
        print("ERROR: could not start MuJoCo sim."); sys.exit(1)

    # open out_csv for append (write header if new)
    new_file = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0
    fout = open(args.out_csv, 'a', newline='')
    writer = csv.DictWriter(fout, fieldnames=RESULT_KEYS)
    if new_file:
        writer.writeheader(); fout.flush()

    all_rows = list(existing)
    t_start = time.time()
    for i, r in enumerate(todo):
        clip = abspath(r['path'])
        name = os.path.basename(clip)
        tag = f"eval_{args.seed}_{i:05d}"
        elapsed = time.time() - t_start
        eta = (elapsed / max(i, 1)) * (len(todo) - i) / 3600.0
        print(f"[{i+1}/{len(todo)}] {r['category'][:18]:18s} {name[:40]:40s} "
              f"(ETA {eta:.1f}h)", flush=True)
        m = None
        try:
            # health check: sim must be alive (restart it if it died mid-run)
            if not sim_running():
                print("   [!] sim died — restarting...", flush=True)
                ensure_sim()
            logdir = run_clip(clip, tag)
            m = M.compute_metrics(logdir) if logdir else None

            # The robot must START the episode standing. The sim is not reset
            # between clips, so a previous fall leaves it on the floor and the
            # next clip measures nothing useful -> restart the sim and retry.
            needs_reset = (m is None) or (not m.get('clean_start')) \
                or (not m.get('saw_mode2'))
            if needs_reset:
                why = ("no metrics" if m is None
                       else ("robot not standing at start "
                             f"(settle tilt {m.get('settle_max_tilt_deg')})"
                             if not m.get('clean_start') else "stream not tracked"))
                print(f"   [!] {why} — resetting sim and retrying", flush=True)
                restart_sim()
                logdir = run_clip(clip, tag)
                m = M.compute_metrics(logdir) if logdir else None
        except Exception as e:
            print(f"   [!] failed: {e}", flush=True)
        if m is None:
            # record a failure row so we don't retry endlessly
            m = {k: '' for k in RESULT_KEYS}
            m['non_fall'] = 0
        m['category'] = r['category']; m['clip'] = name
        writer.writerow({k: m.get(k, '') for k in RESULT_KEYS}); fout.flush()
        all_rows.append({k: str(m.get(k, '')) for k in RESULT_KEYS})
        # tidy per-clip logs to conserve disk
        os.system(f"rm -rf {os.path.join(DEPLOY, 'logs', tag)}")

        # If this episode ended with the robot on the floor, reset the sim now
        # so the NEXT clip starts from a standing robot.
        try:
            if float(m.get('final_tilt_deg') or 0) > 30.0:
                print("   [reset] robot ended fallen — restarting sim", flush=True)
                restart_sim()
        except (TypeError, ValueError):
            pass

        if (i + 1) % 10 == 0:
            aggregate(all_rows, args.per_category, args.report)

    fout.close()
    rep = aggregate(all_rows, args.per_category, args.report)
    print("\n" + rep)
    print(f"\nSaved: {args.out_csv}, {args.report}")


if __name__ == "__main__":
    main()
