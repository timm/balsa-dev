#!/usr/bin/env python3 -B
"""regtab: the regression twin of the paper's mean-AUC / ranked-best table.

Scores are rel = MAE / MAE(predict the mean), so 1.0 is no better than
the mean and SMALLER IS BETTER -- the opposite of AUC.  rank() therefore
walks the treatments in ASCENDING median.  Everything else is the rig's
own rule, ported from balsa_eg.yue so the paper's two tables pool by one
rule: two treatments are the same when their mean rel values are within
`prudence`, or when Cliff's delta over the fold values is at most 0.195
and a KS test is at most its 0.05 critical value.

TabICL is capped by rows (see tabicl_reg.py), so it has no score on the
biggest sets.  Rather than average four treatments over four different
set lists, the table is built over the sets TabICL finished, and one
extra row reports the other three over everything.

usage: python3 regtab.py runs/reg56.csv runs/sereg.csv
"""
import csv, os, sys, math, bisect as B, statistics as S, collections

CLIFF, KS, PRU = 0.195, 1.36, 0.01
ARMS = [("b1", "balsa1"), ("lgbm", "lgbm"), ("b30", "balsa30"),
        ("TabICL", "tabicl")]


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
    "d = [(name, folds)...].  Smaller rel is better, so ascending median."
    d = sorted(d, key=lambda kv: S.median(kv[1]))
    r, pool, out = 1, None, {}
    for nm, v in d:
        if pool is not None and not same(pool, v): r, pool = r + 1, None
        pool = (pool or []) + list(v); out[nm] = r
    return out


def vals(D, s, a):
    k = f"rel_{a}"
    return [float(r[k]) for r in D[s] if r.get(k, "") != ""]


def block(D, group, arms):
    """Means per treatment over the sets it ran; ranked-best over the sets
    where all of `arms` ran, since a rank is only meaningful when every
    treatment is present to be ranked."""
    mu = {k: (S.fmean(v) if (v := [x for s in group for x in vals(D, s, c)])
              else None) for k, c in arms}
    both = [s for s in group if all(vals(D, s, c) for _, c in arms)]
    win = dict.fromkeys([k for k, _ in arms], 0)
    for s in both:
        rk = rank([(k, vals(D, s, c)) for k, c in arms])
        for k in win:
            if rk[k] == 1: win[k] += 1
    return mu, win, len(both)


def line(D, name, group, arms=None):
    """One table row. The three cheap treatments are scored over the whole
    group; TabICL over the subset it could run, whose size is the last
    column, so no group vanishes just because TabICL was capped there."""
    arms = arms or ARMS
    mu, _, _ = block(D, group, arms[:3])          # means: all sets
    tmu, _, nt = block(D, group, arms)            # TabICL and its subset
    _, win, _ = block(D, group, arms[:3] if nt == 0 else arms)
    mu["TabICL"] = tmu["TabICL"]
    seen = [v for v in mu.values() if v is not None]
    lo, hi = min(seen), max(win.values())
    num = lambda k: ("---" if mu[k] is None else
                     (r"\cellcolor{gray!25}" if mu[k] == lo else "")
                     + f"{mu[k]:.2f}")
    cnt = lambda k: ("---" if k not in win else
                     (r"\cellcolor{gray!25}" if win[k] == hi else "")
                     + f"{win[k]}")
    cells = [num(k) for k, _ in arms] + [cnt(k) for k, _ in arms]
    return (f"{name} & {len(group)} & " + " & ".join(cells)
            + f" & {nt or '---'}" + r" \\", mu, win)


def groups(noxse, se):
    "The panels of the two figures, in their plotting order."
    MOOT = os.path.expanduser("~/gits/moot/regression")
    n, numeric = {}, {}
    for s in noxse:
        L = [l for l in open(f"{MOOT}/{s}.csv") if l.strip()]
        head = [h.strip() for h in L[0].split(",")]
        n[s] = len(L) - 1
        x = [h for h in head
             if not h.rstrip("+-!").endswith("X") and h[-1] not in "+-!"]
        numeric[s] = all(h[0].isupper() for h in x)
    G = [("numeric",      [s for s in noxse if n[s] < 1000 and numeric[s]]),
         ("mixed",        [s for s in noxse if n[s] < 1000 and not numeric[s]]),
         ("large, mixed", [s for s in noxse if n[s] >= 1000])]
    for d in sorted({s.split("/")[0] for s in se}):
        G.append((d, [s for s in se if s.split("/")[0] == d]))
    return G


def main():
    D = collections.defaultdict(list)
    A, B_ = set(), set()
    for path, bag in ((sys.argv[1], A), (sys.argv[2], B_)):
        for r in csv.DictReader(open(path)):
            D[r["data"]].append(r); bag.add(r["data"])
    G = groups(sorted(A), sorted(B_))
    ran = {s for s in D if vals(D, s, "tabicl")}   # the sets TabICL finished

    out, rows = [], []
    for name, g in G:
        tex, mu, win = line(D, name.replace(", ", ",\\,"), sorted(g))
        out.append(tex)
        rows.append((name, len(g), len([s for s in g if s in ran]), mu, win))
    out.append(r"\cmidrule{1-11}")
    tex, mu, win = line(D, "total", sorted(D))
    out.append(tex); rows.append(("total", len(D), len(ran), mu, win))
    print("\n".join(out))

    print("\n# the same numbers, plainly")
    skip = sorted(set(D) - ran)
    for name, k, kk, mu, win in rows:
        m = "  ".join(f"{a}={mu[a]:.2f}" if mu.get(a) is not None
                      else f"{a}=  - " for a, _ in ARMS)
        w = " ".join(f"{a}={win[a]}" if a in win else f"{a}=-"
                     for a, _ in ARMS)
        print(f"{name:14s} {k:3d} sets ({kk:3d} w/ TabICL)  {m}   best: {w}")
    print(f"\n# {len(skip)} sets with no TabICL: {' '.join(skip)}")


if __name__ == "__main__": main()
