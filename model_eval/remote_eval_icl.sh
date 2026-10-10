#!/usr/bin/env bash
# remote_eval_icl.sh -- evaluate checkpoints on the ICL hard set (~/ego_dataset/ICL_hardset) on the two GPU
# servers used for finetuning:
#   franka = franka-lambda@192.168.8.228   (repo ~/GR00T-WholeBodyControl, env miniconda env_isaaclab)
#   ur5e   = ur5e@192.168.8.121            (repo ~/gr00t_ft/GR00T-WholeBodyControl, env miniconda gr00t_ft)
#
# AUTH (not stored here): either `export SSHPASS=...` (needs sshpass) or install keys once with
#   /home/grease/GR00T-WholeBodyControl/gear_sonic/scripts/remote_ctl.sh setup-keys
#
# Commands
#   sync    <server|all>                         rsync ICL_hardset to the server(s)
#   eval    <server|all> <steps> [--run PAT] [--tag T]
#                                                run the eval for each step of the newest matching run
#                                                (steps like 28000,30000 or an absolute /path/model.pt)
#   status  <server|all>                         running evals + latest log lines
#   results <server|all>                         pull metrics_eval.json and print per-set + per-clip tables
#
# Defaults mirror watch_and_eval.sh:  franka --run mixpico_v3_sit_local   ur5e --run mixpico_v4
#
# Example:
#   SSHPASS=... model_eval/remote_eval_icl.sh sync all
#   SSHPASS=... model_eval/remote_eval_icl.sh eval all 28000,30000
#   SSHPASS=... model_eval/remote_eval_icl.sh results all
set -uo pipefail

LOCAL_EGO=${LOCAL_EGO:-/home/grease/ego_dataset}
DATASET=ICL_hardset
PULL_DIR=${PULL_DIR:-/home/grease/GR00T-WholeBodyControl/logs_eval/_remote_icl}
SERVERS="franka ur5e"
die() { echo "ERROR: $*" >&2; exit 1; }

load_server() {
  case "$1" in
    franka) HOST=franka-lambda@192.168.8.228; REPO=/home/franka-lambda/GR00T-WholeBodyControl
            PYBIN=/home/franka-lambda/miniconda3/envs/env_isaaclab/bin/python; EGO=/home/franka-lambda/ego_dataset
            EXTRA_ENV=""; DEFAULT_RUN=mixpico_v3_sit_local ;;
    ur5e)   HOST=ur5e@192.168.8.121; REPO=/home/ur5e/gr00t_ft/GR00T-WholeBodyControl
            PYBIN=/home/ur5e/miniconda3/envs/gr00t_ft/bin/python; EGO=/home/ur5e/gr00t_ft/ego_dataset
            EXTRA_ENV="OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y WANDB_MODE=offline"; DEFAULT_RUN=mixpico_v4 ;;
    *) die "unknown server '$1' (franka | ur5e | all)" ;;
  esac
}
targets() { case "${1:-all}" in all) echo "$SERVERS" ;; franka|ur5e) echo "$1" ;; *) die "unknown server '$1'" ;; esac; }

SSH_OPTS=(-o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
rssh()  { if [ -n "${SSHPASS:-}" ]; then sshpass -e ssh "${SSH_OPTS[@]}" "$HOST" "$@"; else ssh -o BatchMode=yes "${SSH_OPTS[@]}" "$HOST" "$@"; fi; }
rscp()  { if [ -n "${SSHPASS:-}" ]; then sshpass -e scp -q "${SSH_OPTS[@]}" "$1" "$HOST:$2"; else scp -q -o BatchMode=yes "${SSH_OPTS[@]}" "$1" "$HOST:$2"; fi; }
rpull() { if [ -n "${SSHPASS:-}" ]; then sshpass -e scp -q "${SSH_OPTS[@]}" "$HOST:$1" "$2"; else scp -q -o BatchMode=yes "${SSH_OPTS[@]}" "$HOST:$1" "$2"; fi; }
rrsync(){ local e="ssh ${SSH_OPTS[*]}"; if [ -n "${SSHPASS:-}" ]; then sshpass -e rsync -e "$e" "$@"; else rsync -e "ssh -o BatchMode=yes ${SSH_OPTS[*]}" "$@"; fi; }

cmd_sync() {
  [ -d "$LOCAL_EGO/$DATASET/robot" ] || die "$LOCAL_EGO/$DATASET not found (run data_process/build_icl_hardset.py)"
  local s
  for s in $(targets "${1:-all}"); do
    load_server "$s"
    echo "== sync $DATASET -> $s ($(du -shL "$LOCAL_EGO/$DATASET" | cut -f1))"
    rssh "mkdir -p '$EGO/$DATASET'" || die "cannot reach $s (set SSHPASS or run remote_ctl.sh setup-keys)"
    rrsync -aL --partial --info=progress2 "$LOCAL_EGO/$DATASET/" "$HOST:$EGO/$DATASET/"
    echo "   remote clips: smpl $(rssh "ls '$EGO/$DATASET/smpl' | wc -l")  robot $(rssh "ls '$EGO/$DATASET/robot' | wc -l")"
  done
}

cmd_eval() {
  local server=${1:?usage: eval <server|all> <steps|/abs/path.pt> [--run PAT] [--tag T]}; local steps=${2:?}; shift 2
  local runpat="" tag=""
  while [ $# -gt 0 ]; do case "$1" in --run) runpat=$2; shift 2 ;; --tag) tag=$2; shift 2 ;; *) die "unknown option $1" ;; esac; done
  local s
  for s in $(targets "$server"); do
    load_server "$s"
    local pat=${runpat:-$DEFAULT_RUN} t=${tag:-icl_$s}
    rssh "test -d '$EGO/$DATASET/robot'" || die "$s: $EGO/$DATASET missing -- run: $0 sync $s"
    local items
    items=$(rssh bash -s -- "$REPO" "$pat" "$steps" "$t" <<'EOF'
REPO=$1; PAT=$2; STEPS=$3; TAG=$4
RUN=$(ls -dt "$REPO"/logs_rl/TRL_G1_Track/manager/universal_token/all_modes/*-2026* 2>/dev/null | grep -v -E "STALLED|INVALID|smoke" | grep -E "$PAT" | head -1)
for s in ${STEPS//,/ }; do
  case "$s" in
    /*.pt) echo "$(basename "$s" .pt)|$s|$TAG" ;;
    *) n=$(printf '%06d' $((10#$s))); echo "s$n|$RUN/model_step_$n.pt|${TAG}_$n" ;;
  esac
done
EOF
    )
    [ -n "$items" ] || die "$s: could not resolve checkpoints"
    echo "$items" | sed "s/^/[$s] checkpoint: /"
    local ts runner name; ts=$(date +%Y%m%d_%H%M%S); runner=$(mktemp); name="eval_icl_${s}_${ts}"
    {
      echo '#!/usr/bin/env bash'
      echo "cd '$REPO'"
      [ -n "$EXTRA_ENV" ] && echo "export $EXTRA_ENV"
      echo "export PY='$PYBIN' DIR='$EGO/$DATASET'"
      printf 'ITEMS=(\n'; while IFS= read -r l; do printf '  %q\n' "$l"; done <<< "$items"; printf ')\n'
      cat <<'RUNNER'
mkdir -p logs_train
N=$(ls "$DIR"/robot/*.pkl | wc -l)
for item in "${ITEMS[@]}"; do
  IFS='|' read -r _ CKPT TAG <<< "$item"
  [ -f "$CKPT" ] || { echo "MISSING $CKPT"; continue; }
  for d in logs_eval/*EVAL_icl_hardset_${TAG}; do [ -f "$d/metrics_eval.json" ] && { echo "$TAG already done"; continue 2; }; done
  "$PY" gear_sonic/eval_agent_trl.py checkpoint="$CKPT" +num_envs="$N" +headless=true +run_once=true \
    +eval_callbacks=im_eval '+eval_output_dir=${eval_log_dir}' \
    ++manager_env.observations.policy.enable_corruption=False ++manager_env.observations.tokenizer.enable_corruption=False \
    +manager_env/terminations=tracking/eval \
    +manager_env.commands.motion.motion_lib_cfg.motion_file="$DIR/robot" \
    +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="$DIR/smpl" \
    ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions="$N" \
    eval_name="EVAL_icl_hardset_${TAG}" > "logs_train/eval_${TAG}_icl.log" 2>&1
  ok=no; for d in logs_eval/*EVAL_icl_hardset_${TAG}; do [ -f "$d/metrics_eval.json" ] && ok=yes; done
  echo "$TAG icl_hardset done=$ok"
done
echo EVAL_RUNNER_DONE
RUNNER
    } > "$runner"
    rssh "mkdir -p '$REPO/logs_train'" && rscp "$runner" "$REPO/logs_train/$name.sh" && rm -f "$runner"
    rssh "cd '$REPO' && nohup setsid bash logs_train/$name.sh > logs_train/$name.log 2>&1 < /dev/null & echo started"
    echo "[$s] launched: $REPO/logs_train/$name.log   (check: $0 status $s)"
  done
}

cmd_status() {
  local s
  for s in $(targets "${1:-all}"); do
    load_server "$s"
    echo "================ $s"
    rssh "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; ps -eo pid,etime,args | grep -E 'eval_agent_trl|eval_icl_' | grep -v grep | cut -c1-150; for f in \$(ls -t '$REPO'/logs_train/eval_icl_*.log 2>/dev/null | head -1); do echo \$f; tail -3 \$f; done; ls -d '$REPO'/logs_eval/*EVAL_icl_hardset_* 2>/dev/null | tail -6"
  done
}

cmd_results() {
  mkdir -p "$PULL_DIR"
  local s
  for s in $(targets "${1:-all}"); do
    load_server "$s"
    local d n
    for d in $(rssh "ls -d '$REPO'/logs_eval/*EVAL_icl_hardset_* 2>/dev/null"); do
      n=$(basename "$d"); mkdir -p "$PULL_DIR/$s"
      rpull "$d/metrics_eval.json" "$PULL_DIR/$s/$n.json" 2>/dev/null && echo "pulled $s/$n"
    done
  done
  python3 - "$PULL_DIR" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
runs = {}
for f in sorted(glob.glob(f"{root}/*/*.json")):
    j = json.load(open(f)); name = os.path.basename(f)[:-5].split("EVAL_icl_hardset_")[-1]
    srv = f.split("/")[-2]; runs[f"{srv}:{name}"] = j
if not runs:
    sys.exit("no results pulled yet")
print(f"\n{'run':44s} {'success':>8s} {'progress':>9s} {'mpjpe_l':>8s} {'mpjpe_g':>8s}")
for k, j in runs.items():
    print(f"{k:44s} {j.get('eval/success/success_rate', float('nan')):8.3f} {j.get('eval/success/progress_rate', float('nan')):9.3f} "
          f"{j.get('eval/all/mpjpe_l', float('nan')):8.1f} {j.get('eval/all/mpjpe_g', float('nan')):8.1f}")
# per-clip table (all runs side by side): progress, X = failed
a = {k: j["eval/all_metrics_dict"] for k, j in runs.items()}
names = sorted({n for m in a.values() for n in m["motion_keys"]})
print("\nper-clip  progress (X = terminated early)")
print(f"{'clip':56s} " + " ".join(f"{k.split(':')[1][-14:]:>14s}" for k in a))
for n in names:
    row = []
    for k, m in a.items():
        if n in m["motion_keys"]:
            i = m["motion_keys"].index(n); row.append(f"{float(m['progress'][i]):6.2f}{'X' if m['terminated'][i] else ' '}")
        else:
            row.append("   -   ")
    print(f"{n[:56]:56s} " + " ".join(f"{r:>14s}" for r in row))
PY
}

case "${1:-help}" in
  sync) shift; cmd_sync "$@" ;;
  eval) shift; cmd_eval "$@" ;;
  status) shift; cmd_status "$@" ;;
  results) shift; cmd_results "$@" ;;
  *) sed -n 2,24p "$0" ;;
esac
