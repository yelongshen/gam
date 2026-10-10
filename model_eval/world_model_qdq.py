#!/usr/bin/env python3
"""Robot world model: predict next-step joint state (q, dq) from a short history of real robot data.

    (s_{t-k+1}, a_{t-k+1}), ..., (s_t, a_t)  ->  s_{t+1} = (q_{t+1}, dq_{t+1})

s_t = (q, dq) of the 29 joints, a_t = commanded target q_target (from the logged policy action),
plus the IMU gravity direction / angular velocity and the commanded PD gain scale per joint
(1.5 on the ankle pitch for the 1001 runs that used scaled kp,kd).  The network predicts the
CHANGE (dq, ddq*dt) so the "nothing changes" predictor is the zero output.

Data: ONLY runs dated after 2026-09-19 (0922, 0924, 1001), from deploy_constants.list_runs()
(9 known-bad runs already excluded).  Validation is held out BY RUN and, by default, by session
type, so it never shares a run with training:

    train : 0922 (4 runs) + 1001 runs that used NOMINAL gains (6 runs)
    val A : 0924 (2 runs)                  -> a different session, nominal gains
    val B : 1001 runs with kp,kd x1.5 on ankle pitch (4 runs) -> a different hardware setting

The final-epoch model is evaluated (no checkpoint selection on the validation sets, so there is
no validation leakage).

Reports, per validation set and per joint group (legs / waist / arms), in physical units:
  * one-step MAE and RMSE of q (deg) and dq (rad/s) for: persistence, ridge regression, MLP
  * multi-step OPEN-LOOP rollout error of q (deg) and dq (rad/s) at 5, 10, 25 steps
    (0.1, 0.2, 0.5 s), feeding the model its own predictions and the REAL q_target, IMU, gains.

Usage:
  .venv_sim/bin/python model_eval/world_model_qdq.py
  .venv_sim/bin/python model_eval/world_model_qdq.py --epochs 80 --hist 5 --hidden 256 --out model_eval/world_model_qdq.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim2real"))
from deploy_constants import CONTROL_DT, DEFAULT_ANGLES, JOINT_NAMES, KD, KP, list_runs, load_csv, load_run  # noqa: E402

AFTER = 20260919
SCALED_1001 = {"084808", "085817", "085443", "090302"}       # kp,kd x1.5 on joints 4 and 10
ANKLE_PITCH = [4, 10]
GROUPS = {"legs": list(range(0, 12)), "waist": [12, 13, 14], "arms": list(range(15, 29)),
          "ankle_pitch": ANKLE_PITCH, "all": list(range(29))}


def run_date(path):
    top = path.split("g1_robot_data/")[1].split("/")[0]
    digits = "".join(c for c in top if c.isdigit())[-4:]
    return int("2026" + digits) if len(digits) == 4 else 0


def run_id(path):
    return os.path.basename(path.rstrip("/")).split("_")[-1]


def gravity_body(quat_wxyz):
    """Gravity direction in the body frame from the IMU quaternion (w,x,y,z). Shape (T,3)."""
    w, x, y, z = quat_wxyz.T
    # third column of R^T applied to world -z : R^T [0,0,-1]
    gx = -2 * (x * z - w * y)
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    return np.stack([gx, gy, gz], 1)


def load_one(path):
    """-> dict of aligned arrays for one run (moving window only) or None."""
    d = load_run(path)
    q, dq, qt = d["q"], d["dq"], d["q_target"]
    if any(np.isnan(v).any() for v in (q, dq, qt)):
        return None
    mv = np.abs(dq).max(axis=1) > 0.05
    if mv.sum() < 200:
        return None
    lo, hi = int(np.argmax(mv)), len(mv) - int(np.argmax(mv[::-1]))
    try:
        bq, _ = load_csv(os.path.join(path, "base_quat.csv"), "base_q")
        bw, _ = load_csv(os.path.join(path, "base_ang_vel.csv"), "base_w")
    except Exception:  # noqa: BLE001
        return None
    n = min(len(q), len(bq), len(bw))
    hi = min(hi, n)
    if hi - lo < 100:
        return None
    sl = slice(lo, hi)
    gain = np.ones(29)
    if run_date(path) == 20261001 and run_id(path) in SCALED_1001:
        gain[ANKLE_PITCH] = 1.5
    T = hi - lo
    return dict(q=q[sl], dq=dq[sl], qt=qt[sl], g=gravity_body(bq[sl]), w=bw[sl],
                gain=np.tile(gain, (T, 1)), name=f"{run_date(path)}/{run_id(path)}")


def split_runs(min_date=0):
    """Validation sets are FIXED (same runs whatever `min_date`); `min_date` only restricts TRAINING runs
    (run date >= min_date; 0 = every clean run, including the ones before 2026-09-19)."""
    train, val_a, val_b = [], [], []
    for r in list_runs():
        dt, rid = run_date(r), run_id(r)
        if dt == 20260924:
            val_a.append(r)
        elif dt == 20261001 and rid in SCALED_1001:
            val_b.append(r)
        elif dt >= min_date:
            train.append(r)
    return train, val_a, val_b


def load_set(paths):
    out = []
    for p in paths:
        r = load_one(p)
        if r is not None:
            out.append(r)
    return out


# ----------------------------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------------------------
def feat_from_hist(qh, dqh, qth, g, w, gain):
    """qh,dqh,qth: (B,k,29), index 0 = current step t, 1 = t-1 ...; g,w: (B,3); gain (B,29).

    The commanded PD gain scale enters ONLY through the PD torque feature
    pd = kp*gain*(q_target - q) - kd*gain*dq  (the actuator law verified on the logs), never as a raw
    input: the training runs all use nominal gains, so a raw gain input would be constant in training
    and meaningless (and destabilising) on the gain-scaled validation runs.
    """
    B = qh.shape[0]
    kp = torch.as_tensor(KP, dtype=qh.dtype)
    kd = torch.as_tensor(KD, dtype=qh.dtype)
    rel_q = (qh - torch.as_tensor(DEFAULT_ANGLES, dtype=qh.dtype)).reshape(B, -1)
    err = (qth - qh)
    pd = kp * gain[:, None, :] * err - kd * gain[:, None, :] * dqh
    return torch.cat([rel_q, dqh.reshape(B, -1), err.reshape(B, -1), pd.reshape(B, -1), g, w], 1)


def make_windows(runs, k):
    """Returns tensors for all valid t (k-1 <= t <= T-2) over all runs."""
    Q, DQ, QT, G, W, GN, TQ, TDQ, RID, TIDX = [], [], [], [], [], [], [], [], [], []
    for ri, r in enumerate(runs):
        T = len(r["q"])
        ts = np.arange(k - 1, T - 1)
        idx = ts[:, None] - np.arange(k)[None, :]               # (n,k) newest first
        Q.append(r["q"][idx]); DQ.append(r["dq"][idx]); QT.append(r["qt"][idx])
        G.append(r["g"][ts]); W.append(r["w"][ts]); GN.append(r["gain"][ts])
        TQ.append(r["q"][ts + 1] - r["q"][ts]); TDQ.append(r["dq"][ts + 1] - r["dq"][ts])
        RID.append(np.full(len(ts), ri)); TIDX.append(ts)
    f = lambda a: torch.as_tensor(np.concatenate(a), dtype=torch.float32)
    return dict(q=f(Q), dq=f(DQ), qt=f(QT), g=f(G), w=f(W), gain=f(GN), dq_t=f(TQ), ddq_t=f(TDQ),
                rid=np.concatenate(RID), tidx=np.concatenate(TIDX))


def windows_to_X(win):
    return feat_from_hist(win["q"], win["dq"], win["qt"], win["g"], win["w"], win["gain"])


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, din, dout, hidden=256, layers=3, p=0.1):
        super().__init__()
        mods, d = [], din
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p)]
            d = hidden
        mods.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class Norm:
    def __init__(self, x):
        self.m = x.mean(0, keepdim=True)
        self.s = x.std(0, keepdim=True).clamp_min(1e-6)

    def __call__(self, x):
        return (x - self.m) / self.s

    def inv(self, x):
        return x * self.s + self.m


class WorldModel:
    """next state = current state + predicted (dq-step, ddq-step); ridge or MLP head."""

    def __init__(self, kind, k, hidden=256, layers=3, dropout=0.1, device="cpu"):
        self.kind, self.k, self.device = kind, k, device
        self.hidden, self.layers, self.dropout = hidden, layers, dropout

    def fit(self, win, epochs, lr, wd, bs, seed, verbose=True, val=None):
        torch.manual_seed(seed)
        X = windows_to_X(win)
        Y = torch.cat([win["dq_t"], win["ddq_t"]], 1)               # (N, 58): [dq_t(q change), ddq (dq change)]
        self.xn, self.yn = Norm(X), Norm(Y)
        Xn, Yn = self.xn(X), self.yn(Y)
        if self.kind == "ridge":
            lam = 10.0
            A = Xn.T @ Xn + lam * torch.eye(Xn.shape[1])
            self.W = torch.linalg.solve(A, Xn.T @ Yn)
            return
        self.net = MLP(Xn.shape[1], Yn.shape[1], self.hidden, self.layers, self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        Xn, Yn = Xn.to(self.device), Yn.to(self.device)
        N = len(Xn)
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(N, device=self.device)
            tot = 0.0
            for i in range(0, N, bs):
                b = perm[i:i + bs]
                loss = ((self.net(Xn[b]) - Yn[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(b)
            sched.step()
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                msg = f"    [mlp] epoch {ep:3d}  train loss {tot / N:.4f}"
                if val is not None:                      # monitoring only; the final epoch is used, no selection
                    msg += "  | val(normalised) " + "  ".join(f"{n} {self.norm_loss(w):.4f}" for n, w in val.items())
                print(msg, flush=True)

    @torch.no_grad()
    def norm_loss(self, win):
        X = self.xn(windows_to_X(win))
        Y = self.yn(torch.cat([win["dq_t"], win["ddq_t"]], 1))
        return float(((self._head(X) - Y) ** 2).mean())

    def _head(self, Xn):
        if self.kind == "ridge":
            return Xn @ self.W
        self.net.eval()
        return self.net(Xn.to(self.device)).cpu()

    @torch.no_grad()
    def step(self, qh, dqh, qth, g, w, gain):
        """one-step prediction of (q_next, dq_next) from the history tensors."""
        X = self.xn(feat_from_hist(qh, dqh, qth, g, w, gain))
        Y = self.yn.inv(self._head(X))
        return qh[:, 0] + Y[:, :29], dqh[:, 0] + Y[:, 29:]


class Persist:
    kind = "persist"

    @torch.no_grad()
    def step(self, qh, dqh, qth, g, w, gain):
        return qh[:, 0], dqh[:, 0]


# ----------------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------------
def group_stats(err_q, err_dq):
    """err: (N,29) in rad, rad/s -> {group: (q_mae_deg, q_rmse_deg, dq_mae, dq_rmse)}"""
    out = {}
    for g, js in GROUPS.items():
        eq, ed = np.degrees(err_q[:, js]), err_dq[:, js]
        out[g] = (float(np.abs(eq).mean()), float(np.sqrt((eq ** 2).mean())),
                  float(np.abs(ed).mean()), float(np.sqrt((ed ** 2).mean())))
    return out


def one_step_eval(model, win):
    qn, dqn = model.step(win["q"], win["dq"], win["qt"], win["g"], win["w"], win["gain"])
    q_true = win["q"][:, 0] + win["dq_t"]
    dq_true = win["dq"][:, 0] + win["ddq_t"]
    return group_stats((qn - q_true).numpy(), (dqn - dq_true).numpy()), (qn - q_true).numpy(), (dqn - dq_true).numpy()


@torch.no_grad()
def rollout_eval(model, runs, k, horizons=(5, 10, 25), n_starts=300, seed=0):
    """Open-loop: start from REAL history, feed predictions back, real q_target / IMU / gains."""
    rng = np.random.default_rng(seed)
    H = max(horizons)
    starts = []
    for ri, r in enumerate(runs):
        T = len(r["q"])
        cand = np.arange(k - 1, T - 1 - H)
        if len(cand):
            sel = rng.choice(cand, size=min(n_starts // len(runs) + 1, len(cand)), replace=False)
            starts += [(ri, int(t)) for t in sel]
    res = {h: ([], []) for h in horizons}
    for ri, t0 in starts:
        r = runs[ri]
        idx = t0 - np.arange(k)
        qh = torch.as_tensor(r["q"][idx], dtype=torch.float32)[None]
        dqh = torch.as_tensor(r["dq"][idx], dtype=torch.float32)[None]
        qth = torch.as_tensor(r["qt"][idx], dtype=torch.float32)[None]
        for h in range(1, H + 1):
            t = t0 + h - 1
            g = torch.as_tensor(r["g"][t], dtype=torch.float32)[None]
            w = torch.as_tensor(r["w"][t], dtype=torch.float32)[None]
            gn = torch.as_tensor(r["gain"][t], dtype=torch.float32)[None]
            qn, dqn = model.step(qh, dqh, qth, g, w, gn)
            qh = torch.cat([qn[:, None], qh[:, :-1]], 1)
            dqh = torch.cat([dqn[:, None], dqh[:, :-1]], 1)
            qth = torch.cat([torch.as_tensor(r["qt"][t + 1], dtype=torch.float32)[None, None], qth[:, :-1]], 1)
            if h in res:
                res[h][0].append((qn[0].numpy() - r["q"][t + 1]))
                res[h][1].append((dqn[0].numpy() - r["dq"][t + 1]))
    return {h: group_stats(np.asarray(res[h][0]), np.asarray(res[h][1])) for h in horizons}, len(starts)


def fmt_group_table(title, stats_by_model, groups=("legs", "waist", "arms", "ankle_pitch", "all")):
    print(f"\n{title}")
    print(f"  {'model':10s}{'group':13s}{'q MAE':>8s}{'q RMSE':>8s}{'dq MAE':>8s}{'dq RMSE':>9s}   (q in deg, dq in rad/s)")
    for mname, st in stats_by_model.items():
        for g in groups:
            a, b, c, d = st[g]
            print(f"  {mname:10s}{g:13s}{a:8.3f}{b:8.3f}{c:8.3f}{d:9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", type=int, default=5, help="history length k")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-starts", type=int, default=300, help="rollout start points per validation set")
    ap.add_argument("--out", default=None, help="save the MLP checkpoint + metrics here")
    ap.add_argument("--min-date", type=int, default=0,
                    help="train only on runs dated >= YYYYMMDD (0 = all dates; use 20260920 for 'after 0919 only'). "
                         "Validation runs are fixed and unaffected.")
    a = ap.parse_args()

    tr_p, va_p, vb_p = split_runs(a.min_date)
    print(f"training runs with date >= {a.min_date or 'ANY'}:  train {len(tr_p)}  val A (0924) {len(va_p)}  "
          f"val B (1001 gain x1.5) {len(vb_p)}")
    train, val_a, val_b = load_set(tr_p), load_set(va_p), load_set(vb_p)
    print("  train:", [r["name"] for r in train])
    print("  val A:", [r["name"] for r in val_a])
    print("  val B:", [r["name"] for r in val_b])
    k = a.hist
    wtr = make_windows(train, k)
    vals = {"valA_0924": make_windows(val_a, k), "valB_1001_x1.5": make_windows(val_b, k)}
    vruns = {"valA_0924": val_a, "valB_1001_x1.5": val_b}
    print(f"windows: train {len(wtr['q'])}  " + "  ".join(f"{n} {len(w['q'])}" for n, w in vals.items()))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ridge = WorldModel("ridge", k); ridge.fit(wtr, 0, 0, 0, 0, a.seed)
    mlp = WorldModel("mlp", k, a.hidden, a.layers, a.dropout, dev)
    print("training MLP ...", flush=True)
    mlp.fit(wtr, a.epochs, a.lr, a.wd, a.bs, a.seed, verbose=True, val=vals)
    models = {"persist": Persist(), "ridge": ridge, "mlp": mlp}

    summary = {}
    ns = 0
    for vname, win in vals.items():
        st = {n: one_step_eval(m, win)[0] for n, m in models.items()}
        fmt_group_table(f"=== ONE-STEP, {vname} ({len(win['q'])} steps) ===", st)
        summary[vname] = {"one_step": st}
        ro = {}
        for n, m in models.items():
            ro[n], ns = rollout_eval(m, vruns[vname], k, n_starts=a.n_starts, seed=a.seed)
        summary[vname]["rollout"] = {n: {str(h): v for h, v in r.items()} for n, r in ro.items()}
        print(f"\n=== OPEN-LOOP ROLLOUT, {vname} ({ns} start points) ===")
        for h in (5, 10, 25):
            print(f"  horizon {h} steps ({h * CONTROL_DT:.2f} s):  q RMSE deg / dq RMSE rad/s, groups legs | waist | arms | ankle_pitch")
            for n in models:
                s = ro[n][h]
                print(f"    {n:8s}" + " | ".join(f"{g}: {s[g][1]:6.2f} / {s[g][3]:5.2f}" for g in ("legs", "waist", "arms", "ankle_pitch")))

    # per-joint one-step table on val A and B for the MLP vs persistence
    for vname, win in vals.items():
        _, eq_m, ed_m = one_step_eval(mlp, win)
        _, eq_p, ed_p = one_step_eval(models["persist"], win)
        print(f"\n=== per joint, {vname}: one-step MAE  (persist -> MLP) ===")
        print(f"  {'joint':22s}{'q MAE deg':>20s}{'dq MAE rad/s':>22s}")
        for j in range(29):
            print(f"  {JOINT_NAMES[j]:22s}{np.degrees(np.abs(eq_p[:, j]).mean()):9.3f} ->{np.degrees(np.abs(eq_m[:, j]).mean()):7.3f}"
                  f"{np.abs(ed_p[:, j]).mean():12.3f} ->{np.abs(ed_m[:, j]).mean():7.3f}")

    if a.out:
        torch.save({"state_dict": mlp.net.state_dict(), "xn": (mlp.xn.m, mlp.xn.s), "yn": (mlp.yn.m, mlp.yn.s),
                    "args": vars(a), "train_runs": [r["name"] for r in train]}, a.out)
        json.dump(summary, open(os.path.splitext(a.out)[0] + "_metrics.json", "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
