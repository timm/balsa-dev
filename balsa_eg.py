# vim: set ts=2 sw=2 sts=2 et cc=75:
"""The dialog: experiments, stats and printing; balsa.py is the model.

usage: pypy3 balsa_eg.py [data names...]     5x2 xval, 30 trees
       pypy3 balsa_eg.py --replay F.dat      rerun F's recorded folds
       pypy3 balsa_eg.py --reg [sets...]     regression corpus, MAE
       pypy3 balsa_eg.py --opt sets...       optimize: label best 5
"""
# (c) 2026 Tim Menzies <timm@ieee.org>, MIT license
import os, sys, time
from math import sqrt
from balsa import (the, srand, shuffle, bisect, csv, Tbl, Col, adds,
                   Forest, aucs)

kl_at = lambda tb: tb.cols.klass.at

# Stats. same() needs all three tests AND a mean gap worth caring about.
def median(xs):
  ys = sorted(xs)
  return ys[len(ys) // 2]

def cohen(num1, num2):
  sp = sqrt(((num1.n - 1) * num1.div()**2 + (num2.n - 1) * num2.div()**2)
            / max(1, num1.n + num2.n - 2))
  return abs(num1.mu - num2.mu) / (sp + 1e-32)

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
  num1, num2 = adds(xs), adds(ys)
  if prudence is None: prudence = the.prudence
  if abs(num1.mu - num2.mu) < prudence: return True
  return (cliffs(xs, ys) <= 0.195 and ks(xs, ys) <= 1.36)

def rank(d):
  out, pool, r = {}, None, 1
  for k, xs in sorted(d.items(), key=lambda kv: median(kv[1])):
    if pool and not same(pool, xs): r, pool = r + 1, None
    pool = (pool or []) + list(xs)
    out[k] = r
  return out

# Rigs. Row ids are 0-based csv row numbers, so any other tool can
# replay the exact same folds.
def xval(rows, fn, m=5, n=2):
  for _ in range(m):
    u = shuffle(rows)
    for kf in range(n):
      test  = [r for j, r in enumerate(u) if j % n == kf]
      train = [r for j, r in enumerate(u) if j % n != kf]
      fn(train, test)

def rowids(tb):
  n = len(tb.rows)
  return {id(row): n - 1 - j for j, row in enumerate(tb.rows)}

def moot(sub, f):
  root = os.getenv("MOOT" + sub.upper(), "~/gits/moot/" + sub)
  return "%s/%s.csv" % (root, f)

def ens(sets):
  print("#n30 ")
  for f in sets:
    tb = Tbl(csv(moot("classify", f)))
    at = kl_at(tb)
    y = lambda row: row.labels[at]
    ids = rowids(tb)
    def fold(train, test):
      t0 = time.process_time()
      a = aucs(Forest(tb, train, y).votes(test, y))
      print("%s|%.3f |%.3f |%s " % (f, a, time.process_time() - t0,
            " ".join(str(ids[id(r)]) for r in test)), flush=True)
    xval(tb.rows, fold)

def replay(dat):
  print("#1:n1 n30 ")
  tb = v = name = None
  for line in open(dat):
    if line.startswith("#"): continue
    f, _, _, idstr = line.strip().split("|")
    if f != name:
      name = f
      tb = Tbl(csv(moot("classify", f)))
      n = len(tb.rows)
      v = {n - 1 - j: row for j, row in enumerate(tb.rows)}
    at = kl_at(tb)
    y = lambda row: row.labels[at]
    test = [v[int(s)] for s in idstr.split()]
    inh = set(map(id, test))
    train = [row for row in tb.rows if id(row) not in inh]
    out, secs = [], []
    for trees, rf in ((1, False), (30, True)):
      the.trees, the.rf = trees, rf
      t0 = time.process_time()
      out.append(aucs(Forest(tb, train, y).votes(test, y)))
      secs.append(time.process_time() - t0)
    print("%s|%.3f %.3f |%.3f %.3f |%s " % (f, out[0], out[1],
          secs[0], secs[1], idstr), flush=True)

def mae(want, got):
  return sum(abs(w - g) for w, g in zip(want, got)) / max(1, len(want))

def goalcol(tb):
  g = tb.cols.klass
  if g is None:
    g = min(tb.cols.y, key=lambda c: c.at)   # first goal, file order
  return g

def reg(sets):
  print("#1:reg n30:reg mean ")
  for f in sets:
    tb = Tbl(csv(moot("regression", f)))
    at = goalcol(tb).at
    y = lambda row: row[at]
    ids = rowids(tb)
    def fold(train, test):
      keep = [r for r in test if y(r) != "?"]
      want = [y(r) for r in keep]
      ok = [y(r) for r in train if y(r) != "?"]
      mu = sum(ok) / max(1, len(ok))
      maes, secs = [], []
      for trees, rf in ((1, False), (30, True)):
        the.trees, the.rf = trees, rf
        t0 = time.process_time()
        trees_ = Forest(tb, train, y, Col)
        maes.append(mae(want, [trees_.predict(r) for r in keep]))
        secs.append(time.process_time() - t0)
      maes.append(mae(want, [mu] * len(keep)))
      print("%s|%.4g %.4g %.4g |%.3f %.3f |%s " % (f, maes[0],
            maes[1], maes[2], secs[0], secs[1],
            " ".join(str(ids[id(r)]) for r in test)), flush=True)
    xval(tb.rows, fold)

def opt(sets):
  print("#1:opt n30:opt rand ")
  for f in sets:
    root = os.getenv("MOOTOPT", "~/gits/moot/optimize")
    tb = Tbl(csv("%s/%s.csv" % (root, f)))
    for row in tb.rows: row.d2h = tb.disty(row)
    y = lambda row: row.d2h
    lo = min(r.d2h for r in tb.rows)
    mu = sum(r.d2h for r in tb.rows) / len(tb.rows)
    score = lambda got: 100 * (1 - (got - lo) / (mu - lo + 1e-32))
    best5 = lambda rows: min(r.d2h for r in rows[:5])
    ids = rowids(tb)
    def fold(train, test):
      out, secs = [], []
      for trees, rf in ((1, False), (30, True)):
        the.trees, the.rf = trees, rf
        t0 = time.process_time()
        trees_ = Forest(tb, train, y, Col)
        guess = sorted(test, key=lambda r: trees_.predict(r))
        out.append(score(best5(guess)))
        secs.append(time.process_time() - t0)
      out.append(score(best5(shuffle(test))))
      print("%s|%.2f %.2f %.2f |%.3f %.3f |%s " % (f, out[0], out[1],
            out[2], secs[0], secs[1],
            " ".join(str(ids[id(r)]) for r in test)), flush=True)
    xval(tb.rows, fold)

ENS = ["balance.scale", "blood-transfusion-service-center", "breast.w",
  "diabetes", "ecoli", "glass", "ionosphere", "sonar", "vehicle",
  "wdbc", "vote", "soybean", "tic-tac-toe", "german", "car",
  "solar.flare2", "primary.tumor", "breast.cancer", "spect.test",
  "audiology", "credit.a", "eucalyptus", "ilpd", "heart.c",
  "hepatitis", "colic", "autos", "cylinder.bands", "credit.g", "cmc",
  "hypothyroid", "sick", "Internet-Advertisements", "churn",
  "COMPAS53", "nomao", "bank-marketing", "electricity", "adult"]

REG = ["2dplanes", "abalone", "ailerons", "auto93", "autohorse",
  "autompg", "autoprice", "bank32nh", "bank8FM", "baskball",
  "bodyfat", "bolts", "breasttumor", "cal.housing", "cholesterol",
  "cleveland", "cloud", "cpu.act", "cpu", "cpu.small", "detroit",
  "diabetes.numeric", "echomonths", "elevators", "elusage",
  "fishcatch", "fried", "fruitfly", "gascons", "house16H", "house8L",
  "housing", "hungarian", "kin8nm", "longley", "lowbwt",
  "machine.cpu", "mbagrade", "meta", "mv", "pbc", "pharynx", "pol",
  "pollution", "puma32H", "puma8NH", "pwlinear", "pyrim", "quake",
  "schlvote", "sensory", "servo", "sleep", "stock", "strike",
  "triazines", "veteran", "vineyard", "wisconsin"]

if __name__ == "__main__":
  srand(the.seed)
  a = sys.argv[1:]
  if a and a[0] == "--replay": replay(a[1])
  elif a and a[0] == "--reg":  reg(a[1:] or REG)
  elif a and a[0] == "--opt":  opt(a[1:])
  else:                        ens(a or ENS)
