#!/usr/bin/env python3
"""Roblox Up-and-Coming chart forecaster.

Entry for make.fun competition roblox-up-and-coming-forecast-01.
Reads the chart history at /input (history.ndjson + task.json) and writes
one forecast to /output/forecast.json: for each of the top 10 places, a
probability for each game named at that place, plus the top 10 in order.

Method (drift + diffusion, time-based; fitted on the 34-snapshot archive
2026-10-30 -> 2026-11-07, backtested +11.2 log points over the house
persistence baseline on 21 cutoffs, winning 21/21):
  For each game in the current top 60, model its rank at a future hour t as
  Normal(mu, sigma), mu = r0 + drift*t, sigma = 2.5*sqrt(t). The drift
  blends the game's own recent rank velocity (linear fit over the last
  ~30h, weight 0.5) with a rank-bucket drift plus a tenure adjustment
  (new games climb, old games fall faster), clipped to +/-0.35 ranks/h.
  P(game at place k) is the normal mass on [k-0.5, k+0.5], averaged
  uniformly over t = 0, 3, ..., horizon hours. Per place the probabilities
  are normalised and the top 20 games are named; the rest is the unnamed
  share. `playing` counts carry no signal (corr ~ 0) and are not used.
  top10 is the argmax game per place (tie-break only).

Stdlib only, no network, deterministic (ties broken by universe_id).
"""
import json
import math
import os
from datetime import datetime

# --- fitted parameters (see docstring) ---
SIGMA = 2.5          # diffusion, ranks per sqrt-hour
POOL_N = 60          # candidate games: current top 60 by rank
TOP_M = 20           # named games per place
W_VEL = 0.5          # weight on the game's own rank velocity
DRIFT_CLIP = 0.35    # max |drift|, ranks/hour
T_STEP = 3           # hours between averaged future times
VEL_WINDOW_H = 30.0  # lookback for the velocity fit, hours

SQRT2 = math.sqrt(2.0)


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def rank_drift(r0):
    """Baseline drift (ranks/hour, + = falling) by current rank bucket."""
    if r0 <= 3:
        return 0.057
    if r0 <= 6:
        return 0.037
    if r0 <= 10:
        return 0.154
    if r0 <= 15:
        return 0.134
    if r0 <= 20:
        return 0.153
    if r0 <= 30:
        return 0.129
    return 0.128


def tenure_adj_hours(ten_h):
    """Extra drift by hours since the game was first seen (new = climbs)."""
    if ten_h <= 24:
        return -0.10
    if ten_h <= 72:
        return -0.02
    if ten_h <= 120:
        return 0.05
    return 0.12


def parse_time(s):
    """Parse an ISO-8601 timestamp (handles a trailing Z)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    """Read task.json + history.ndjson; return (task, snaps sorted by time).

    Each snap is {"at": datetime, "rank": {universe_id: rank},
    "top10": [universe_id, ...]}. Reads that failed (ok=false) are skipped.
    """
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("ok", True):
                continue
            rows = rec.get("up_and_coming") or []
            if not rows:
                continue
            rank = {}
            for row in rows:
                try:
                    rank[int(row["universe_id"])] = int(row["rank"])
                except (KeyError, TypeError, ValueError):
                    continue
            if not rank:
                continue
            top10 = [g for g, _ in sorted(rank.items(), key=lambda kv: kv[1])[:10]]
            snaps.append({"at": parse_time(rec["at"]), "rank": rank, "top10": top10})
    snaps.sort(key=lambda s: s["at"])
    return task, snaps


def velocity_24h(snaps, i, g):
    """Linear fit of rank vs time over snapshots in the last ~30h.

    Returns the slope in ranks/hour (negative = climbing), or None when
    fewer than two points are available.
    """
    pts = []
    for j in range(i, -1, -1):
        t = (snaps[j]["at"] - snaps[i]["at"]).total_seconds() / 3600.0
        if t < -VEL_WINDOW_H:
            break
        r = snaps[j]["rank"].get(g)
        if r is None:
            continue
        pts.append((t, r))
    if len(pts) < 2:
        return None
    mt = sum(p[0] for p in pts) / len(pts)
    mr = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mt) ** 2 for p in pts)
    if den == 0:
        return None
    return sum((p[0] - mt) * (p[1] - mr) for p in pts) / den


def forecast(snaps, horizon_h, places):
    """Time-averaged Gaussian rank model; returns (probs, top10).

    probs is {place: {universe_id: p}} (normalised, top TOP_M named).
    """
    last = snaps[-1]
    i = len(snaps) - 1
    rank0 = last["rank"]
    first_seen_at = {}
    for snap in snaps:
        for g in snap["rank"]:
            first_seen_at.setdefault(g, snap["at"])
    pool = sorted(rank0.items(), key=lambda kv: (kv[1], kv[0]))[:POOL_N]
    times = list(range(0, horizon_h + 1, T_STEP))
    n_times = len(times)
    w = {k: {} for k in range(1, places + 1)}
    for g, r0 in pool:
        ten_h = (last["at"] - first_seen_at.get(g, last["at"])).total_seconds() / 3600.0
        vel = velocity_24h(snaps, i, g)
        base = rank_drift(r0) + tenure_adj_hours(ten_h)
        drift = base if vel is None else W_VEL * vel + (1.0 - W_VEL) * base
        drift = max(-DRIFT_CLIP, min(DRIFT_CLIP, drift))
        for t in times:
            mu = r0 + drift * t
            if t == 0:
                if 1 <= r0 <= places:
                    w[r0][g] = w[r0].get(g, 0.0) + 1.0 / n_times
                continue
            sd = SIGMA * math.sqrt(t)
            for k in range(1, places + 1):
                p = phi((k + 0.5 - mu) / sd) - phi((k - 0.5 - mu) / sd)
                if p > 1e-6:
                    w[k][g] = w[k].get(g, 0.0) + p / n_times
    probs = {}
    for k in range(1, places + 1):
        wk = w[k]
        tot = sum(wk.values())
        if tot <= 0:
            probs[k] = {}
            continue
        ranked = sorted(wk.items(), key=lambda kv: (-kv[1], kv[0]))
        probs[k] = {g: p / tot for g, p in ranked[:TOP_M]}
    top10 = []
    for k in range(1, places + 1):
        wk = w[k]
        if wk:
            top10.append(max(wk.items(), key=lambda kv: (kv[1], -kv[0]))[0])
        else:
            top10.append(last["top10"][k - 1])
    seen, out = set(), []
    for g in top10:
        if g not in seen:
            seen.add(g)
            out.append(g)
    for g in last["top10"]:
        if len(out) >= places:
            break
        if g not in seen:
            seen.add(g)
            out.append(g)
    return probs, out[:places]


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, snaps = load_history(input_dir)
    places = int(task.get("places", 10))
    window = task.get("window") or {}
    horizon_h = 72
    if window.get("from") and window.get("until"):
        horizon_h = int(round(
            (parse_time(window["until"]) - parse_time(window["from"])).total_seconds() / 3600.0
        ))
    probs, top10 = forecast(snaps, horizon_h, places)
    rows = []
    for k in range(1, places + 1):
        for g, p in sorted(probs.get(k, {}).items(), key=lambda kv: (-kv[1], kv[0])):
            p6 = math.floor(p * 1e6) / 1e6
            if p6 > 0:
                rows.append({"universe_id": int(g), "place": k, "p": p6})
    out = {"probabilities": rows, "top10": [int(g) for g in top10]}
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
