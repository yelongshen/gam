"""Check alignment of the 4 order-paired 20261001 episodes: slide the STREAMED clip's
retargeted dof over the executed q of the whole streamed span; report best offset/rms
vs the window the evaluator used, and vs the best other clip."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_runs_0922 import (robust_csv, streamed_ranges, sliding_ssd, resample_to_50hz,
                            load_robot_refs, Q_COL0, NUM_DOF)
R = os.path.expanduser('~/g1_robot_data/g1_run_1001')
d = json.load(open('sim2real/online_eval_results_20261001.json'))
refs, _ = load_robot_refs(os.path.expanduser('~/ego_dataset/eval_subset/robot'))
print(type(refs), len(refs))
for e in d['episodes']:
    run = os.path.join(R, e['run'])
    hdr, arr = robust_csv(os.path.join(run, 'q.csv'))
    t = arr[:, hdr.index('time_ms')]
    q = arr[:, Q_COL0:Q_COL0+NUM_DOF]
    used = np.searchsorted(t, e['episode_def']['t0_ms'])
    rg = refs if isinstance(refs, dict) else dict(refs)
    res = {}
    for clip, r in rg.items():
        r = np.asarray(r)
        if r.shape[0] > len(q): continue
        ssd = sliding_ssd(q, r)
        if ssd is None: continue
        o = int(np.argmin(ssd)); res[clip] = (np.degrees(np.sqrt(ssd[o]/(r.shape[0]*NUM_DOF))), o)
    me = res.get(e['clip'])
    best = sorted(res.items(), key=lambda kv: kv[1][0])[:3]
    print(e['run'][-6:], e['clip'][:20], 'used_off', used, 'ep_def L', round(e['imitation']['mpjpe_l'],1),
          '| own clip best off/rms', me and (me[1], round(me[0],2)), '| top3', [(k[:22], round(v[0],2)) for k, v in best])
