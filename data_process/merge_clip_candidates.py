#!/usr/bin/env python3
"""merge_clip_candidates.py — turn the overlapping candidate list produced by
`detect_action_clips.py` into a clean, NON-OVERLAPPING clip list.

Why this is needed
------------------
`detect_action_clips.py` emits two things, and both can overlap:

  * candidates  -- raw active spans. Long spans are split into equal chunks, so
                   consecutive pieces abut, and gap-bridging can produce spans
                   that partially cover each other.
  * selected    -- the top-N per label, each SNAPPED to a multiple of
                   --frame_base (100). Snapping moves edges by up to +/-2 s and
                   routinely makes neighbours overlap, e.g. [6100,6600) and
                   [6600,7100), or [2500,2800) and [2700,3100).

For building a clip set you want disjoint ranges. This merges any spans that
overlap (or sit within --join-gap of each other), then assigns each merged
region a single label by *duration-weighted vote* over its constituents, and
optionally re-splits anything longer than --max-dur.

Usage
-----
    .venv_sim/bin/python data_process/merge_clip_candidates.py \
        --report data_analysis/pico_20260924/clips_20260924_173612.txt

    # all sessions at once, writing merged lists next to the reports
    .venv_sim/bin/python data_process/merge_clip_candidates.py \
        --report data_analysis/pico_20260924/clips_*.txt --write
"""
import argparse
import collections
import glob
import os
import re

CAND_RE = re.compile(
    r'\[\s*(\d+),\s*(\d+)\)\s+dur=\s*([\d.]+)s\s+label=(\w+)\s+coarse=(.+)')
SEL_RE = re.compile(r'\[\s*(\d+),\s*(\d+)\)\s+([\d.]+)\s+(\w+)\s+(\S.*?)\s{2,}')


def parse_report(path):
    txt = open(path).read()
    cands = []
    if 'All ' in txt:
        for a, b, d, lab, coarse in CAND_RE.findall(txt.split('All ')[1]):
            cands.append(dict(start=int(a), end=int(b), label=lab,
                              coarse=coarse.strip()))
    sels = []
    if 'SELECTED CLIPS' in txt:
        block = txt.split('SELECTED CLIPS')[1].split('All ')[0]
        for a, b, d, lab, coarse in SEL_RE.findall(block):
            sels.append(dict(start=int(a), end=int(b), label=lab,
                             coarse=coarse.strip()))
    return cands, sels


def relabel(m):
    """(Re)compute the duration-weighted label vote for a merged region."""
    votes = collections.Counter()
    cvotes = collections.Counter()
    for p in m['parts']:
        w = p['end'] - p['start']
        votes[p['label']] += w
        cvotes[p['coarse']] += w
    m['label'] = votes.most_common(1)[0][0]
    m['coarse'] = cvotes.most_common(1)[0][0]
    m['labels'] = dict(votes)
    m['n_parts'] = len(m['parts'])
    return m


def absorb_short(merged, min_frames, max_gap):
    """Fold clips shorter than `min_frames` into a neighbour within `max_gap`.

    Short fragments are usually one motion that the energy threshold chopped up
    (a brief dip below the percentile mid-stride). Repeatedly attach the
    shortest such fragment to whichever neighbour is closer, until nothing
    short is left within reach.
    """
    out = [dict(m) for m in merged]
    while True:
        cands = [i for i, m in enumerate(out) if (m['end'] - m['start']) < min_frames]
        if not cands:
            break
        moved = False
        # shortest first, so the most fragmentary pieces get absorbed earliest
        for i in sorted(cands, key=lambda i: out[i]['end'] - out[i]['start']):
            gap_prev = out[i]['start'] - out[i - 1]['end'] if i > 0 else None
            gap_next = out[i + 1]['start'] - out[i]['end'] if i < len(out) - 1 else None
            opts = [(g, j) for g, j in ((gap_prev, i - 1), (gap_next, i + 1))
                    if g is not None and g <= max_gap]
            if not opts:
                continue
            _, j = min(opts, key=lambda gj: gj[0])
            tgt = out[j]
            tgt['start'] = min(tgt['start'], out[i]['start'])
            tgt['end'] = max(tgt['end'], out[i]['end'])
            tgt['parts'] = tgt['parts'] + out[i]['parts']
            tgt['absorbed'] = tgt.get('absorbed', 0) + 1
            relabel(tgt)
            del out[i]
            moved = True
            break
        if not moved:
            break                     # remaining short clips are all isolated
    return out


def merge(spans, join_gap=0):
    """Merge overlapping / near-touching spans. Label by duration-weighted vote."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s['start'], s['end']))
    out = [dict(start=spans[0]['start'], end=spans[0]['end'], parts=[spans[0]])]
    for s in spans[1:]:
        cur = out[-1]
        if s['start'] <= cur['end'] + join_gap:          # overlap or touch
            cur['end'] = max(cur['end'], s['end'])
            cur['parts'].append(s)
        else:
            out.append(dict(start=s['start'], end=s['end'], parts=[s]))

    for m in out:
        relabel(m)
    return out


def split_long(merged, max_frames):
    """Re-split merged regions longer than max_frames into equal-ish pieces."""
    if not max_frames:
        return merged
    out = []
    for m in merged:
        n = m['end'] - m['start']
        if n <= max_frames:
            out.append(m)
            continue
        k = -(-n // max_frames)                 # ceil
        step = -(-n // k)
        for c0 in range(m['start'], m['end'], step):
            c1 = min(m['end'], c0 + step)
            out.append({**m, 'start': c0, 'end': c1, 'split_of': (m['start'], m['end'])})
    return out


def split_by_label(merged, min_frames=0):
    """Cut each merged region back apart where the LABEL changes.

    A long merged region is often one continuous activity period containing
    several distinct actions (e.g. waving -> walking -> agile). Group the
    constituent parts into consecutive runs of the same label, and place the
    cut at the midpoint of the overlap (or gap) between adjacent runs so the
    result is still exactly disjoint.

    Runs shorter than `min_frames` are folded into the neighbouring run rather
    than emitted as slivers.
    """
    out = []
    for m in merged:
        parts = sorted(m['parts'], key=lambda p: (p['start'], p['end']))
        runs = []
        for p in parts:
            if runs and runs[-1]['label'] == p['label']:
                runs[-1]['parts'].append(p)
                runs[-1]['start'] = min(runs[-1]['start'], p['start'])
                runs[-1]['end'] = max(runs[-1]['end'], p['end'])
            else:
                runs.append(dict(start=p['start'], end=p['end'],
                                 label=p['label'], coarse=p['coarse'], parts=[p]))
        if len(runs) == 1:
            out.append(m)
            continue

        # Resolve overlaps between adjacent runs at the midpoint FIRST, so the
        # sliver test below sees each run's final extent (a run can shrink a
        # lot here: [2400,2500) overlapped on both sides ends up 1 s wide).
        for i in range(len(runs) - 1):
            a, b = runs[i], runs[i + 1]
            if b['start'] < a['end']:
                mid = (b['start'] + a['end']) // 2
                a['end'], b['start'] = mid, mid

        # Fold sliver runs into the longer neighbour, extending its boundary.
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for i, r in enumerate(runs):
                if (r['end'] - r['start']) >= min_frames:
                    continue
                prev_len = (runs[i - 1]['end'] - runs[i - 1]['start']) if i > 0 else -1
                next_len = (runs[i + 1]['end'] - runs[i + 1]['start']) if i < len(runs) - 1 else -1
                j = i - 1 if prev_len >= next_len else i + 1
                runs[j]['parts'] += r['parts']
                runs[j]['start'] = min(runs[j]['start'], r['start'])
                runs[j]['end'] = max(runs[j]['end'], r['end'])
                del runs[i]
                changed = True
                break

        for r in runs:
            r['n_parts'] = len(r['parts'])
            r['labels'] = {r['label']: r['end'] - r['start']}
            r['split_from'] = (m['start'], m['end'])
            out.append(r)
    return out


def render(session, merged, fps, src_n, src_label):
    lines = [f"\n===== {session}   {src_label}: {src_n} -> {len(merged)} merged, "
             f"non-overlapping"]
    lines.append(f"{'#':>3} {'chunk range':>17} {'t (s)':>15} {'dur':>7}  "
                 f"{'label':10} {'parts':>5}  composition")
    total = 0
    for i, m in enumerate(merged, 1):
        n = m['end'] - m['start']
        total += n
        comp = ', '.join(f"{k}:{v/fps:.1f}s" for k, v in
                         sorted(m['labels'].items(), key=lambda kv: -kv[1]))
        tag = '' if m.get('split_of') is None else ' (split)'
        lines.append(f"{i:3d} [{m['start']:6d},{m['end']:6d}) "
                     f"{m['start']/fps:6.1f}-{m['end']/fps:6.1f} {n/fps:6.1f}s  "
                     f"{m['label']:10} {m['n_parts']:5d}  {comp}{tag}")
    lines.append(f"    total {total/fps:.1f}s across {len(merged)} disjoint clips")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', nargs='+', required=True,
                    help='clips_<session>.txt report(s) from detect_action_clips.py')
    ap.add_argument('--fps', type=float, default=50.0)
    ap.add_argument('--join-gap', type=int, default=0,
                    help='also merge spans separated by <= this many chunks')
    ap.add_argument('--absorb-short', type=float, default=None,
                    help='fold clips shorter than this many seconds into a '
                         'neighbour (see --absorb-gap)')
    ap.add_argument('--absorb-gap', type=int, default=200,
                    help='max chunk gap for --absorb-short (default 200 = 4 s @50Hz)')
    ap.add_argument('--split-by-label', action='store_true',
                    help='cut merged regions back apart where the label changes')
    ap.add_argument('--min-run', type=float, default=1.5,
                    help='with --split-by-label, runs shorter than this many '
                         'seconds are folded into a neighbour (default 1.5)')
    ap.add_argument('--max-dur', type=float, default=None,
                    help='re-split merged clips longer than this many seconds')
    ap.add_argument('--source', choices=['candidates', 'selected'],
                    default='candidates', help='which list to merge')
    ap.add_argument('--write', action='store_true',
                    help='write merged_<session>.txt next to each report')
    args = ap.parse_args()

    paths = []
    for r in args.report:
        paths.extend(sorted(glob.glob(r)))

    for p in paths:
        session = os.path.basename(p)[len('clips_'):-len('.txt')]
        cands, sels = parse_report(p)
        spans = cands if args.source == 'candidates' else sels
        merged = merge(spans, args.join_gap)
        if args.absorb_short:
            merged = absorb_short(merged, int(args.absorb_short * args.fps),
                                  args.absorb_gap)
        if args.split_by_label:
            merged = split_by_label(merged, int(args.min_run * args.fps))
        merged = split_long(merged, int(args.max_dur * args.fps) if args.max_dur else None)
        text = render(session, merged, args.fps, len(spans), args.source)
        print(text)
        if args.write:
            out = os.path.join(os.path.dirname(p), f'merged_{session}.txt')
            with open(out, 'w') as fh:
                fh.write(text + '\n')
            print(f'    [written] {out}')


if __name__ == '__main__':
    main()
