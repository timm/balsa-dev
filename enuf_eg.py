# vim: set ts=2 sw=2 sts=2 et cc=75:
"""The rig behind the paper: statistics, and the replay of recorded
folds. Every number in a figure or table comes from one of the three
replay modes below; gen.py drew those folds in the first place.

usage: pypy3 enuf_eg.py --replay    F.dat   classification AUCs
       pypy3 enuf_eg.py --regreplay F.dat   regression MAE
       pypy3 enuf_eg.py --optreplay F.dat   optimize: label best 5
"""
# (c) 2026 Tim Menzies <timm@ieee.org>, MIT license
import os, sys, time
from math import sqrt
from enuf import the, srand, shuffle, bisect, csv, Tbl, Col, Forest, aucs

kl_at   = lambda tb: tb.cols.klass.at
mae     = lambda want, got: (sum(abs(w - g) for w, g in zip(want, got))
                             / max(1, len(want)))

def goalcol(tb):
  "The class, else the first goal in file order (for regression)."
  return tb.cols.klass or min(tb.cols.y, key=lambda c: c.at)

def mootdir(sub):
  return os.getenv("MOOT" + sub.upper(), "~/gits/moot/" + sub)

# Statistics. same() pools two treatments only if their means are too
# close to care about, OR Cliff's delta and a KS test both say so.
def median(xs):
  ys = sorted(xs)
  return ys[len(ys) // 2]

def cliffs(xs, ys):
  d = sum(bisect(ys, x) - (len(ys) - bisect(ys, x, True)) for x in xs)
  return abs(d) / (len(xs) * len(ys))

def ks(xs, ys):
  n, m = len(xs), len(ys)
  d = max(abs(bisect(xs, v, True) / n - bisect(ys, v, True) / m)
          for v in xs + ys)
  return d / sqrt((n + m) / (n * m))

def same(xs, ys, prudence=None):
  xs, ys = sorted(xs), sorted(ys)
  if prudence is None: prudence = the.prudence
  if abs(sum(xs)/len(xs) - sum(ys)/len(ys)) < prudence: return True
  return cliffs(xs, ys) <= 0.195 and ks(xs, ys) <= 1.36

def rank(d):
  out, pool, r = {}, None, 1
  for k, xs in sorted(d.items(), key=lambda kv: median(kv[1])):
    if pool and not same(pool, xs): r, pool = r + 1, None
    pool = (pool or []) + list(xs)
    out[k] = r
  return out

# Replay. A dat line is data|scores|secs|test row ids; the ids make
# every fold rerunnable, so any learner meets the same split.
def datroot(dat):
  f = next(l for l in open(dat) if not l.startswith("#"))
  opt = "/" in f.split("|")[0]
  return os.getenv("MOOTOPT" if opt else "MOOTREG",
                   "~/gits/moot/" + ("optimize" if opt else "regression"))

def foldsof(dat, root):
  "Yield (set, tb, test, train, ids) off a dat's recorded row ids."
  tb = name = None
  for line in open(dat):
    if line.startswith("#"): continue
    f, _, _, ids = line.strip().split("|")
    if f != name:
      name, tb = f, Tbl(csv("%s/%s.csv" % (root, f)))
    test = [tb.rows[int(x)] for x in ids.split()]
    inh = set(map(id, test))
    train = [row for row in tb.rows if id(row) not in inh]
    yield f, tb, test, train, ids

def replay(dat):
  print("#1:q16 n30 ")
  for f, tb, test, train, ids in foldsof(dat, mootdir("classify")):
    y = lambda row: row.labels[kl_at(tb)]
    out, secs = [], []
    for trees, rf in ((1, False), (30, True)):
      the.trees, the.rf = trees, rf
      t0 = time.process_time()
      out.append(aucs(Forest(tb, train, y).votes(test, y)))
      secs.append(time.process_time() - t0)
    print("%s|%.3f %.3f |%.3f %.3f |%s " % (f, out[0], out[1],
          secs[0], secs[1], ids), flush=True)

def regreplay(dat):
  print("#1:reg n30:reg mean ")
  for f, tb, test, train, ids in foldsof(dat, datroot(dat)):
    y = lambda row: row[goalcol(tb).at]
    keep = [r for r in test if y(r) != "?"]
    want = [y(r) for r in keep]
    ok = [y(r) for r in train if y(r) != "?"]
    mu = sum(ok) / max(1, len(ok))
    maes, secs = [], []
    for trees, rf in ((1, False), (30, True)):
      the.trees, the.rf = trees, rf
      t0 = time.process_time()
      fr = Forest(tb, train, y, Col)
      maes.append(mae(want, [fr.predict(r) for r in keep]))
      secs.append(time.process_time() - t0)
    maes.append(mae(want, [mu] * len(keep)))
    print("%s|%.4g %.4g %.4g |%.3f %.3f |%s " % (f, maes[0], maes[1],
          maes[2], secs[0], secs[1], ids), flush=True)

def optreplay(dat):
  print("#1:opt n30:opt rand ")
  root = os.getenv("MOOTOPT", "~/gits/moot/optimize")
  seen = {}
  for f, tb, test, train, ids in foldsof(dat, root):
    if f not in seen:
      for row in tb.rows: row.d2h = tb.disty(row)
      lo = min(r.d2h for r in tb.rows)
      mu = sum(r.d2h for r in tb.rows) / len(tb.rows)
      seen[f] = (lo, mu)
    lo, mu = seen[f]
    score = lambda got: 100 * (1 - (got - lo) / (mu - lo + 1e-32))
    best5 = lambda rows: min(r.d2h for r in rows[:5])
    y = lambda row: row.d2h
    out, secs = [], []
    for trees, rf in ((1, False), (30, True)):
      the.trees, the.rf = trees, rf
      t0 = time.process_time()
      fr = Forest(tb, train, y, Col)
      guess = sorted(test, key=lambda r: fr.predict(r))
      out.append(score(best5(guess)))
      secs.append(time.process_time() - t0)
    out.append(score(best5(shuffle(test))))
    print("%s|%.2f %.2f %.2f |%.3f %.3f |%s " % (f, out[0], out[1],
          out[2], secs[0], secs[1], ids), flush=True)

MODES = {"--replay": replay, "--regreplay": regreplay,
         "--optreplay": optreplay}

if __name__ == "__main__":
  srand(the.seed)
  a = sys.argv[1:]
  if len(a) == 2 and a[0] in MODES: MODES[a[0]](a[1])
  else: print(__doc__)
