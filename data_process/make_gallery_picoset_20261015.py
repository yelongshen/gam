"""Build an HTML gallery (index.html) of the paired SMPL | robot GIFs of picoset_20261015, with tags and the event label per clip.
usage: python make_gallery_picoset_20261015.py"""
import csv
import html
import os

VIS = "/home/grease/gam/data_visualization/picoset_20261015_check"
TAGS = "/home/grease/g1_robot_data/pico_raw/20261015_combined_clips/clip_tags.csv"
rows = list(csv.DictReader(open(TAGS)))
have = set(os.listdir(VIS))
cards = []
for r in rows:
    name = f"pico1015_{r['clip']}"
    gif = f"{name}.gif"
    ev = r.get("event", "")
    cls = "event" if ev else ""
    img = f'<img loading="lazy" src="{gif}" width="520">' if gif in have else "<div class=missing>GIF missing</div>"
    cards.append(
        f'<div class="card {cls}"><div class="t"><b>{html.escape(name)}</b> {("<span class=ev>" + html.escape(ev) + "</span>") if ev else ""}</div>'
        f'{img}<div class="m">{r["dur"]} s | path {r["path"]} m | disp {r["disp"]} m | pelvis z {r["pelz_min"]}-{r["pelz_max"]} | tags: {html.escape(r["tags"])}</div></div>'
    )
page = f"""<!doctype html><meta charset="utf-8"><title>picoset_20261015: SMPL | G1 retarget</title>
<style>body{{font-family:sans-serif;background:#111;color:#ddd}} .grid{{display:flex;flex-wrap:wrap;gap:10px}}
.card{{background:#222;padding:6px;border-radius:6px}} .event{{outline:3px solid #f5a623}} .ev{{background:#f5a623;color:#000;padding:1px 6px;border-radius:4px;margin-left:6px}}
.m{{font-size:12px;color:#aaa;max-width:520px}} .t{{margin-bottom:4px}} .missing{{width:520px;height:100px;background:#400}}</style>
<h2>picoset_20261015 - {len(rows)} clips (left: human SMPL, right: retargeted G1). Orange outline = crawl / jump event clips.</h2>
<div class="grid">{''.join(cards)}</div>"""
open(f"{VIS}/index.html", "w").write(page)
print(f"wrote {VIS}/index.html  ({sum((('pico1015_' + r['clip'] + '.gif') in have) for r in rows)}/{len(rows)} gifs present)")
