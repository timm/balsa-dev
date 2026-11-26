#!/usr/bin/env python3
"""The reg60 analogue of the paper's mean-AUC / ranked-best table, on rel_*.

rel_X = mae_X / mae_mean, so 1.0 means "no better than predicting the mean"
and SMALLER IS BETTER -- the opposite of AUC.  rank() therefore walks the
treatments in ascending median, not descending.

Pooling rule is the paper's: two treatments are the same when their mean
rel_ values are within prudence, or when Cliff's delta over the fold values
is at most 0.195 and a KS test at most its 0.05 critical value.
"""
import csv, os, math, bisect as B, statistics as S, collections

CLIFF, KS, PRU = 0.195, 1.36, 0.01
ARMS = [("b1", "balsa1"), ("b30", "balsa30"), ("lgbm", "lgbm")]
SIZES = [("tiny", 0, 200), ("small", 200, 1000),
         ("medium", 1000, 10000), ("large", 10000, 10**9)]

def bis(vs, v, le=False):
    return B.bisect_right(vs, v) if le else B.bisect_left(vs, v)

def cliffs(xs, ys):
    d = sum(bis(ys, x) - (len(ys) - bis(ys, x, True)) for x in xs)
    return abs(d) / (len(xs) * len(ys))

def ks(xs, ys):
    n, m, d = len(xs), len(ys), 0
    for v in (xs, ys):
        for x in v:
            d = max(d, abs(bis(xs, x, True) / n - bis(ys, x, True) / m))
    return d / math.sqrt((n + m) / (n * m))

def same(xs, ys):
    xs, ys = sorted(xs), sorted(ys)
    if abs(S.fmean(xs) - S.fmean(ys)) < PRU: return True
    return cliffs(xs, ys) <= CLIFF and ks(xs, ys) <= KS

def rank(d):
    "d = [(name, folds)...].  Smaller rel_ is better, so ascending median."
    d = sorted(d, key=lambda kv: S.median(kv[1]))
    r, pool, out = 1, None, {}
    for nm, v in d:
        if pool is not None and not same(pool, v): r, pool = r + 1, None
        pool = (pool or []) + list(v); out[nm] = r
    return out

rows = list(csv.DictReader(open(os.path.expanduser(
    "~/gh/balsa-dev/runs/reg60.csv"))))
D = collections.defaultdict(list)
for r in rows: D[r["data"]].append(r)
sets = sorted(D)
M = os.path.expanduser("~/gits/moot/regression")
N = {s: sum(1 for l in open(f"{M}/{s}.csv") if l.strip()) - 1 for s in sets}

def block(group):
    mu = {k: S.fmean([float(r[f"rel_{c}"]) for s in group for r in D[s]])
          for k, c in ARMS}
    win = {k: 0 for k, _ in ARMS}
    for s in group:
        rk = rank([(k, [float(r[f"rel_{c}"]) for r in D[s]]) for k, c in ARMS])
        for k, _ in ARMS:
            if rk[k] == 1: win[k] += 1
    return mu, win

def line(name, group):
    mu, win = block(group)
    lo = min(mu.values()); hi = max(win.values())
    cells  = [(r"\cellcolor{gray!25}" if mu[k] == lo else "") + f"{mu[k]:.2f}"
              for k, _ in ARMS]
    cells += [(r"\cellcolor{gray!25}" if win[k] == hi else "") + f"{win[k]}"
              for k, _ in ARMS]
    return f"{name} & {len(group)} & " + " & ".join(cells) + r" \\"

out = []
for lab, lo, hi in SIZES:
    g = [s for s in sets if lo <= N[s] < hi]
    if g: out.append(line(lab, g))
out.append(r"\cmidrule{1-8}")
out.append(line("total", sets))
print("\n".join(out))
print()
# the same numbers, plainly, for the chat
for lab, lo, hi in SIZES + [("TOTAL", 0, 10**9)]:
    g = [s for s in sets if lo <= N[s] < hi]
    mu, win = block(g)
    print(f"{lab:8s} n={len(g):3d}  " +
          "  ".join(f"{k}={mu[k]:.2f}" for k, _ in ARMS) + "   best: " +
          " ".join(f"{k}={win[k]}" for k, _ in ARMS))
