#!/usr/bin/env python3
"""Pick positions where the ML bot disagreed with the equity move, score
the candidates with the model on the CPU, write a Lua sim script per
position, and (later) fold the sim verdicts into a gallery page.

Steps (each writes into --work):
  select   from mlreads' JSON lines: still-contested positions with a real
           equity gap; dumps each position's top-50 candidates through
           bin/mlreads -dump-* and scores them with the ONNX model; keeps the
           ones where the model's own top pick is the move it played and it
           clearly prefers it to the equity move  -> selected.json
  simscripts  one Lua script per selected position (gen 40, add both moves,
           sim -plies 5 -stop 99)                  -> sim-<id>.lua
  collect  parse the sim outputs                   -> selected.json (+sim)
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(REPO, "data/strategy/default/models/macondo-nn-tf-nwl23s/1/model.onnx")


def load_rows(path):
    return [json.loads(l) for l in open(path)]


def contested(r, args):
    diff = r["ml_score"] - r["opp_score"]
    return (args.bag_min <= r["bag"] <= args.bag_max and abs(diff) <= args.max_diff
            and r["gap"] >= args.min_gap and r["ml_spread"] != 0)


def game_rows(turns_path, game_id, out_path):
    """Extract one game's rows (both halves) from the turn log, with the header."""
    with open(turns_path) as f, open(out_path, "w") as o:
        o.write(f.readline())
        key = f",{game_id},"
        for line in f:
            if key in line:
                o.write(line)


def dump_and_score(r, args, sess, work):
    pid = f"{r['game'].split(':')[1][:10]}-h{r['half']}-t{r['turn']}"
    prefix = os.path.join(work, "dump-" + pid)
    if not os.path.exists(prefix + ".json"):
        one = os.path.join(work, "one-" + pid + ".txt")
        game_rows(args.turns, r["game"], one)
        subprocess.run([os.path.join(REPO, "bin/mlreads"), "-turns", one, "-games", args.games, "-threads", "1",
                        "-dump-game", r["game"], "-dump-half", str(r["half"]), "-dump-turn", str(r["turn"]),
                        "-dump-out", prefix], check=True, capture_output=True)
        os.unlink(one)
    d = json.load(open(prefix + ".json"))
    n = len(d["cands"])
    planes = np.fromfile(prefix + "-planes.bin", dtype=np.float32).reshape(n, 85, 15, 15)
    scal = np.fromfile(prefix + "-scalars.bin", dtype=np.float32).reshape(n, 72)
    names = [i.name for i in sess.get_inputs()]
    val, spread = sess.run(None, {names[0]: planes, names[1]: scal})
    val, spread = val.ravel(), spread.ravel()
    for i, c in enumerate(d["cands"]):
        c["value"], c["spread"] = float(val[i]), float(spread[i])
    d["cands"].sort(key=lambda c: -c["value"])
    return pid, d


def select(args):
    import onnxruntime as ort
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    rows = [r for r in load_rows(args.reads) if contested(r, args)]
    if args.random:
        import random
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.random]
        won = [r for r in rows if r["ml_won"]]
        lost = [r for r in rows if not r["ml_won"]]
    else:
        rows.sort(key=lambda r: -r["gap"])
        won = [r for r in rows if r["ml_won"]][: args.pool]
        lost = [r for r in rows if not r["ml_won"]][: args.pool]
    print(f"{len(rows)} contested disagreements; scoring {len(won)} won + {len(lost)} lost candidates", file=sys.stderr)
    out = []
    for kind, pool in (("won", won), ("lost", lost)):
        for r in pool:
            pid, d = dump_and_score(r, args, sess, args.work)
            cands = d["cands"]
            played = next((c for c in cands if c["played"]), None)
            eqbest = max(cands, key=lambda c: c["equity"])
            if played is None or cands[0] is not played:
                continue  # the replay's candidate set or ranking does not reproduce the bot's pick
            pref = played["value"] - eqbest["value"]
            if pref < args.min_pref or abs(played["value"]) > args.max_abs_value:
                continue
            out.append({**r, "id": pid, "kind": kind, "cands": cands[:12], "played": played, "eqbest": eqbest, "pref": pref})
            print(f"  {kind} {pid}: bag {r['bag']} diff {r['ml_score']-r['opp_score']:+d} {played['move'].strip()} v={played['value']:+.3f} "
                  f"vs {eqbest['move'].strip()} v={eqbest['value']:+.3f} (eq gap {r['gap']:.1f}, pref {pref:+.3f})", file=sys.stderr)
    json.dump(out, open(os.path.join(args.work, "selected.json"), "w"), indent=1)
    print(f"{len(out)} positions selected", file=sys.stderr)


def simscripts(args):
    sel = json.load(open(os.path.join(args.work, "selected.json")))
    for r in sel:
        cgp = r["cgp"]
        ml, eq = r["played"]["move"].strip(), r["eqbest"]["move"].strip()
        lua = f"""local macondo = require("macondo")
macondo.load('cgp {cgp}')
local g = macondo.gen('40')
for _, mv in ipairs({{ {json.dumps(ml)}, {json.dumps(eq)} }}) do
  local key = string.gsub(mv, "%..*$", "")  -- up to the first play-through dot
  if not string.find(g, key, 1, true) then macondo.add(mv) end
end
print("SIMRESULT-BEGIN")
print(macondo.sim("-plies {args.plies} -stop {args.stop} -threads {args.threads}"))
print("SIMRESULT-END")
"""
        open(os.path.join(args.work, f"sim-{r['id']}.lua"), "w").write(lua)
    print(f"{len(sel)} sim scripts in {args.work}", file=sys.stderr)


SIM_LINE = re.compile(r"^\s*(\S+\s+\S+)\s+(\S*)\s+(\d+)\s+([\d.]+)±([\d.]+)\s+(-?[\d.]+)±([\d.]+)")


def parse_sim(text):
    plays = []
    winner = None
    for line in text.splitlines():
        if line.startswith("Sim winner:"):
            winner = line.split(":", 1)[1].strip()
        m = SIM_LINE.match(line)
        if m:
            plays.append({"move": m.group(1), "leave": m.group(2), "score": int(m.group(3)),
                          "win": float(m.group(4)), "win_ci": float(m.group(5)),
                          "equity": float(m.group(6)), "cut": "❌" in line})
    return winner, plays


def collect(args):
    sel = json.load(open(os.path.join(args.work, "selected.json")))
    for r in sel:
        p = os.path.join(args.work, f"sim-{r['id']}.out")
        if not os.path.exists(p):
            continue
        text = open(p).read()
        text = text.split("SIMRESULT-BEGIN", 1)[-1].split("SIMRESULT-END", 1)[0]
        winner, plays = parse_sim(text)
        r["sim"] = {"winner": winner, "plays": plays}
    json.dump(sel, open(os.path.join(args.work, "selected.json"), "w"), indent=1)
    print(f"{sum('sim' in r for r in sel)} of {len(sel)} positions have sim results", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["select", "simscripts", "collect", "page"])
    p.add_argument("--reads", default="reads.jsonl")
    p.add_argument("--turns", default=os.path.join(REPO, "tf-nwl23s-v-hasty-pairs.txt"))
    p.add_argument("--games", default=os.path.join(REPO, "games-tf-nwl23s-v-hasty-pairs.txt"))
    p.add_argument("--work", default="reads-work")
    p.add_argument("--bag-min", type=int, default=15)
    p.add_argument("--bag-max", type=int, default=75)
    p.add_argument("--max-diff", type=int, default=40, help="max |score difference| before the move")
    p.add_argument("--min-gap", type=float, default=6.0, help="min static-equity gap")
    p.add_argument("--pool", type=int, default=30, help="candidates per kind to score")
    p.add_argument("--random", type=int, default=0, help="score a random sample of this many contested disagreements instead of the largest gaps")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--also", default="", help="page: a second work dir shown as a compact table (the largest deviations)")
    p.add_argument("--min-pref", type=float, default=0.02, help="min model preference (value gap) for its pick")
    p.add_argument("--max-abs-value", type=float, default=0.9)
    p.add_argument("--plies", type=int, default=5)
    p.add_argument("--stop", type=int, default=99)
    p.add_argument("--threads", type=int, default=6)
    args = p.parse_args()
    os.makedirs(args.work, exist_ok=True)
    {"select": select, "simscripts": simscripts, "collect": collect, "page": page}[args.step](args)



# ---------------------------------------------------------------------------
# The gallery page
# ---------------------------------------------------------------------------

TWS = {(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)}
DWS = {(i, i) for i in (1, 2, 3, 4, 10, 11, 12, 13)} | {(i, 14 - i) for i in (1, 2, 3, 4, 10, 11, 12, 13)} | {(7, 7)}
TLS = {(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13), (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)}
DLS = {(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14), (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11),
       (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14), (12, 6), (12, 8), (14, 3), (14, 11)}


def parse_board(cgp):
    fen = cgp.split(" ")[0]
    board = [[None] * 15 for _ in range(15)]
    for r, row in enumerate(fen.split("/")):
        c = 0
        num = ""
        for ch in row:
            if ch.isdigit():
                num += ch
                continue
            if num:
                c += int(num)
                num = ""
            board[r][c] = ch
            c += 1
    return board


def move_squares(mv):
    """(squares, letters) a play covers, from ' 14H zAX' notation; empty for exchanges/passes."""
    mv = mv.strip()
    if mv.startswith("(") or mv.lower() == "pass":
        return []
    coords, word = mv.split()
    if coords[0].isdigit():
        i = 0
        while i < len(coords) and coords[i].isdigit():
            i += 1
        r, c, vert = int(coords[:i]) - 1, ord(coords[i]) - 65, False
    else:
        r, c, vert = int(coords[1:]) - 1, ord(coords[0]) - 65, True
    out = []
    for k, ch in enumerate(word):
        rr, cc = (r + k, c) if vert else (r, c + k)
        if ch != ".":
            out.append((rr, cc, ch))
    return out


def board_svg(cgp, ml_sq, eq_sq):
    cell = 26
    size = 15 * cell + 1
    b = parse_board(cgp)
    ml = {(r, c): ch for r, c, ch in ml_sq}
    eq = {(r, c): ch for r, c, ch in eq_sq}
    parts = [f'<svg viewBox="0 0 {size} {size}" class="board" role="img" aria-label="board">']
    for r in range(15):
        for c in range(15):
            x, y = c * cell, r * cell
            cls = "sq"
            if (r, c) in TWS: cls += " tws"
            elif (r, c) in DWS: cls += " dws"
            elif (r, c) in TLS: cls += " tls"
            elif (r, c) in DLS: cls += " dls"
            parts.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{cell}" height="{cell}"/>')
            tile = b[r][c]
            if tile:
                blank = tile.islower()
                parts.append(f'<rect class="tile" x="{x+1.5}" y="{y+1.5}" width="{cell-3}" height="{cell-3}" rx="3"/>')
                parts.append(f'<text class="tl{" bl" if blank else ""}" x="{x+cell/2}" y="{y+cell/2+5}">{tile.upper()}</text>')
    # the two candidate moves: model on top of equity so both stay visible when they overlap
    for sqs, cls in ((eq, "eqm"), (ml, "mlm")):
        for (r, c), ch in sqs.items():
            x, y = c * cell, r * cell
            parts.append(f'<rect class="{cls}" x="{x+2}" y="{y+2}" width="{cell-4}" height="{cell-4}" rx="4"/>')
            parts.append(f'<text class="tl {cls}t{" bl" if ch.islower() else ""}" x="{x+cell/2}" y="{y+cell/2+5}">{ch.upper()}</text>')
    parts.append("</svg>")
    return "".join(parts)


def sim_norm(mv):
    """Sim output writes play-through letters in parentheses; our notation uses dots."""
    return re.sub(r"\(([A-Za-z]+)\)", lambda m: "." * len(m.group(1)), mv).strip()


def verdicts(sel):
    """Attach the sim verdict to each position: head to head between the model's
    play and the equity play (model / equity / close), and whether a third play
    beat both (third)."""
    for r in sel:
        ml, eq = r["played"]["move"].strip(), r["eqbest"]["move"].strip()
        sim = r.get("sim")
        simrows = {}
        v, h2h = "pending", "pending"
        if sim and sim["plays"]:
            for p in sim["plays"]:
                simrows.setdefault(sim_norm(p["move"]), p)  # the first listing has the most iterations
            top = sim["plays"][0]
            a, b = simrows.get(ml), simrows.get(eq)
            if a and b:
                d = a["win"] - b["win"]
                noise = (a["win_ci"] ** 2 + b["win_ci"] ** 2) ** 0.5  # 99% intervals
                h2h = "close" if abs(d) <= noise else "model" if d > 0 else "equity"
                topn = sim_norm(top["move"])
                third = topn not in (ml, eq) and top["win"] - max(a["win"], b["win"]) > top["win_ci"]
                v = "third" if third else h2h
        r["_verdict"], r["_h2h"], r["_ml"], r["_eq"], r["_simrows"] = v, h2h, ml, eq, simrows
    return sel


def card(r):
    ml, eq, sim = r["_ml"], r["_eq"], r.get("sim")
    simrows = r["_simrows"]
    diff = r["ml_score"] - r["opp_score"]
    lead = f"up {diff}" if diff > 0 else f"down {-diff}" if diff < 0 else "level"
    result = f"went on to win by {r['ml_spread']}" if r["ml_spread"] > 0 else f"went on to lose by {-r['ml_spread']}"
    v = r["_verdict"]
    badge = {"model": "sim sides with the model", "equity": "sim sides with equity", "close": "too close to call",
             "third": "a third play is best", "pending": "sim pending"}[v]
    rows = []
    for label, mv, cand, cls in (("model's play", ml, r["played"], "mlm"), ("equity play", eq, r["eqbest"], "eqm")):
        s = simrows.get(mv)
        win = "" if s is None else f"{s['win']:.1f}<span class='ci'>±{s['win_ci']:.1f}</span>"
        rows.append(f"""<tr><th scope="row"><span class="swatch {cls}"></span>{label}</th>
<td class="mv">{mv}</td><td class="n">{cand['score']}</td><td class="n">{cand['equity']:.1f}</td>
<td class="n">{cand['value']:+.3f}</td><td class="n">{win}</td></tr>""")
    third = ""
    if v == "third" and sim:
        p = sim["plays"][0]
        third = f"""<tr><th scope="row">sim's choice</th><td class="mv">{sim_norm(p['move'])}</td><td class="n">{p['score']}</td>
<td class="n"></td><td class="n"></td><td class="n">{p['win']:.1f}<span class='ci'>±{p['win_ci']:.1f}</span></td></tr>"""
    return f"""<article class="pos">
<div class="boardwrap">{board_svg(r['cgp'], move_squares(ml), move_squares(eq))}</div>
<div class="side">
<p class="ctx"><span class="badge {v}">{badge}</span></p>
<h3>Rack <span class="rack">{r['rack']}</span>, {lead}, {r['bag']} in the bag</h3>
<p class="meta">Turn {r['turn']}. Static equity preferred <b>{eq}</b> by {r['gap']:.1f} points; the model preferred <b>{ml}</b> by {r['pref']:+.3f} in value. It {result}.</p>
<div class="tbl"><table>
<thead><tr><th></th><th>play</th><th>score</th><th>equity</th><th>net value</th><th>5-ply sim win%</th></tr></thead>
<tbody>{''.join(rows)}{third}</tbody></table></div>
</div></article>"""


def tally_html(sel, what):
    h = collections.Counter(r["_h2h"] for r in sel)
    t = collections.Counter(r["_verdict"] for r in sel)
    n = sum(1 for r in sel if r["_h2h"] != "pending")
    return f"""<div class="tally"><div><b>{h['model']}</b><span>model's play better</span></div><div><b>{h['equity']}</b><span>equity play better</span></div><div><b>{h['close']}</b><span>too close to call</span></div><div><b>{t['third']}</b><span>a third play beat both</span></div></div>
<p class="lede">{what} Head to head over these {n} positions, the 5-ply sim ranks the model's play above the equity play {h['model']} times and below it {h['equity']} times, with {h['close']} too close to call (the two plays' 99% intervals overlap). In {t['third']} of them a play neither bot chose beat both.</p>"""


def compact_table(sel):
    rows = []
    for r in sorted(sel, key=lambda r: -r["gap"]):
        a, b = r["_simrows"].get(r["_ml"]), r["_simrows"].get(r["_eq"])
        f = lambda p: "" if p is None else f"{p['win']:.1f}±{p['win_ci']:.1f}"
        rows.append(f"""<tr><td>{r['bag']}</td><td>{r['ml_score']-r['opp_score']:+d}</td><td class="mv">{r['_ml']}</td><td class="n">{r['played']['equity']:.0f}</td><td class="n">{f(a)}</td>
<td class="mv">{r['_eq']}</td><td class="n">{r['eqbest']['equity']:.0f}</td><td class="n">{f(b)}</td><td><span class="badge {r['_verdict']}">{r['_verdict'].replace('model','model').replace('close','close')}</span></td></tr>""")
    return f"""<div class="tbl"><table class="compact"><thead><tr><th>bag</th><th>score diff</th><th>model's play</th><th>eq</th><th>sim win%</th><th>equity play</th><th>eq</th><th>sim win%</th><th>verdict</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"""


def page(args):
    sel = verdicts(json.load(open(os.path.join(args.work, "selected.json"))))
    order = {"model": 0, "third": 1, "close": 2, "equity": 3, "pending": 4}
    sel.sort(key=lambda r: (order[r["_verdict"]], -r["gap"]))
    big = verdicts(json.load(open(os.path.join(args.also, "selected.json")))) if args.also else []
    big_section = ""
    if big:
        big_section = f"""<h2>The largest deviations</h2>
{tally_html(big, "A second set, chosen differently: the 35 contested positions with the <em>largest</em> equity gaps among 20,000 games.")}
<p class="lede">These are mostly declined bingos and declined big scores, and the sim usually says the bingo was right. The model's typical deviations are good; its boldest ones are its weakest. In decided games (not shown here) the value head saturates near ±1 and its ranking among plays becomes noise, which is a separate, fixable problem.</p>
{compact_table(big)}"""
    html = f"""<title>Reads Against Equity</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f6f3ec;--ink:#1f1d1a;--ink2:#5c574d;--line:#dcd6c8;--card:#fffdf8;--tile:#f2e4c2;--tileink:#2b2620;
--tws:#e2a79e;--dws:#efc9c0;--tls:#8fb7d6;--dls:#c9dceb;--sq:#ebe6da;--model:#0f8a63;--modelbg:#d9f0e6;--equity:#c9581f;--equitybg:#fbe3d4;--third:#7b6fb5;--thirdbg:#e9e4f7;--pend:#8a857a;}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#181614;--ink:#ece7dc;--ink2:#a9a294;--line:#3a3630;--card:#211e1b;--tile:#e8d8b0;--tileink:#2b2620;--tws:#8a3f37;--dws:#5e3a35;--tls:#2f5878;--dls:#2c3f4f;--sq:#2a2723;--model:#3fc493;--modelbg:#12382b;--equity:#f08a55;--equitybg:#40251a;--third:#b3a7f0;--thirdbg:#2b2740;--pend:#8a857a;}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#181614;--ink:#ece7dc;--ink2:#a9a294;--line:#3a3630;--card:#211e1b;--tile:#e8d8b0;--tileink:#2b2620;--tws:#8a3f37;--dws:#5e3a35;--tls:#2f5878;--dls:#2c3f4f;--sq:#2a2723;--model:#3fc493;--modelbg:#12382b;--equity:#f08a55;--equitybg:#40251a;--third:#b3a7f0;--thirdbg:#2b2740;--pend:#8a857a;}}
body{{background:var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:16px;line-height:1.45;margin:0;padding-block:32px 64px;padding-inline:clamp(16px,4vw,48px);}}
main{{max-width:1080px;margin:0 auto;}}
h1{{font-family:Fraunces,Georgia,serif;font-weight:700;font-size:clamp(28px,4vw,40px);line-height:1.1;margin:0 0 8px;text-wrap:balance;}}
h2{{font-family:Fraunces,Georgia,serif;font-weight:500;font-size:24px;margin:40px 0 12px;}}
h3{{font-family:Fraunces,Georgia,serif;font-weight:500;font-size:19px;margin:6px 0 4px;text-wrap:balance;}}
.lede{{max-width:66ch;color:var(--ink2);margin:0 0 6px;}}
.tally{{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0 8px;}}
.tally div{{padding:10px 14px;border:1px solid var(--line);border-radius:8px;background:var(--card);min-width:120px;}}
.tally b{{display:block;font-family:"IBM Plex Mono",monospace;font-size:26px;font-weight:500;line-height:1.1;}}
.tally span{{color:var(--ink2);font-size:13px;text-transform:uppercase;letter-spacing:.06em;}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;color:var(--ink2);font-size:14px;margin:10px 0 0;}}
.swatch{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px;}}
.swatch.mlm{{background:var(--model);}} .swatch.eqm{{background:var(--equity);}}
.pos{{display:grid;grid-template-columns:minmax(260px,391px) 1fr;gap:20px;align-items:start;padding:18px;margin:14px 0;border:1px solid var(--line);border-radius:10px;background:var(--card);}}
@media (max-width:760px){{.pos{{grid-template-columns:1fr;}}}}
.boardwrap{{max-width:100%;}}
.board{{width:100%;height:auto;display:block;}}
.sq{{fill:var(--sq);stroke:var(--bg);stroke-width:1;}} .tws{{fill:var(--tws);}} .dws{{fill:var(--dws);}} .tls{{fill:var(--tls);}} .dls{{fill:var(--dls);}}
.tile{{fill:var(--tile);}} .tl{{font-family:"IBM Plex Mono",monospace;font-size:14px;font-weight:500;fill:var(--tileink);text-anchor:middle;}} .tl.bl{{font-style:italic;fill:#7a5a1e;}}
.mlm{{fill:var(--model);}} .eqm{{fill:var(--equity);}} .tl.mlmt,.tl.eqmt{{fill:#fff;}} .tl.mlmt.bl,.tl.eqmt.bl{{fill:#fff;font-style:italic;}}
.ctx{{margin:0;}}
.badge{{display:inline-block;font-size:12px;text-transform:uppercase;letter-spacing:.06em;padding:3px 9px;border-radius:999px;font-weight:600;white-space:nowrap;}}
.badge.model{{background:var(--modelbg);color:var(--model);}} .badge.equity{{background:var(--equitybg);color:var(--equity);}} .badge.third{{background:var(--thirdbg);color:var(--third);}} .badge.close{{background:var(--sq);color:var(--ink2);}} .badge.pending{{background:var(--sq);color:var(--pend);}}
.rack{{font-family:"IBM Plex Mono",monospace;letter-spacing:.12em;}}
.meta{{color:var(--ink2);margin:0 0 10px;max-width:60ch;}}
.tbl{{overflow-x:auto;}}
table{{border-collapse:collapse;width:100%;font-size:15px;font-variant-numeric:tabular-nums;}}
th{{text-align:left;color:var(--ink2);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.06em;padding:6px 8px;border-bottom:1px solid var(--line);}}
tbody th{{text-transform:none;font-size:15px;letter-spacing:0;color:var(--ink);white-space:nowrap;}}
td{{padding:7px 8px;border-bottom:1px solid var(--line);}} td.n{{text-align:right;font-family:"IBM Plex Mono",monospace;font-size:14px;white-space:nowrap;}} td.mv{{font-family:"IBM Plex Mono",monospace;white-space:nowrap;}}
table.compact{{font-size:14px;}} table.compact td{{padding:5px 8px;}}
.ci{{color:var(--ink2);font-size:12px;margin-left:2px;}}
footer{{color:var(--ink2);font-size:14px;margin-top:40px;max-width:70ch;}}
</style>
<main>
<h1>Reads against equity</h1>
<p class="lede">What does the value net do differently from static equity, and is it right? These are positions from FastMlBot's 100,000-pair match against HastyBot where the net chose a different play from the top static-equity play. Only contested positions are included: 15 to 75 tiles in the bag, no more than 40 points between the players, and an equity gap of at least 2 points. Each is adjudicated by a 5-ply Monte Carlo simulation run to 99% confidence over the top 40 plays plus the two in question, the strongest evaluator we have.</p>
<h2>A random sample of the net's deviations</h2>
{tally_html(sel, "Drawn at random from 42,822 contested disagreements in 20,000 games.")}
<p class="legend"><span><span class="swatch mlm"></span>the model's play</span><span><span class="swatch eqm"></span>the static-equity play</span><span>Net value is P(win) − P(loss) for the position after the play, from the mover's side.</span></p>
{''.join(card(r) for r in sel)}
{big_section}
<footer>Model: macondo-nn-tf-nwl23s, a 3.7M-parameter transformer value net trained with per-square placement heads, 57.2% paired win rate against HastyBot. Candidates are HastyBot's top 50 plays by static equity; the model plays the one whose resulting position it values highest. The sim is Macondo's <code>sim -plies 5 -stop 99</code>. Whether the game was eventually won is descriptive only: a single game is a noisy verdict, which is why the sim adjudicates.</footer>
</main>
"""
    out = os.path.join(args.work, "reads-gallery.html")
    open(out, "w").write(html)
    print(out, file=sys.stderr)


if __name__ == "__main__":
    main()
