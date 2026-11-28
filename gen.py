# vim: set ts=2 sw=2 sts=2 et cc=75:
"""Draw the folds. These runners make the dat files that enuf_eg.py
later replays; the paper's numbers come from the replay, not here, so
this file is outside the "learner plus rig" line count. Each dat line
is data|scores|secs|test row ids, and the ids are what make it
replayable.

usage: pypy3 gen.py            classification, 39 sets    > nose.dat
       pypy3 gen.py --reg      regression, 59 sets        > reg.dat
       pypy3 gen.py --opt S..  optimize, named sets       > opt.dat
"""
# (c) 2026 Tim Menzies <timm@ieee.org>, MIT license
import sys, time
from enuf import the, srand, shuffle, csv, Tbl, Col, Forest, aucs
from enuf_eg import mootdir, goalcol, mae, kl_at

def xval(rows, fn, m=5, n=2):
  for _ in range(m):
    u = shuffle(rows)
    for kf in range(n):
      test  = [r for j, r in enumerate(u) if j % n == kf]
      train = [r for j, r in enumerate(u) if j % n != kf]
      fn(train, test)

def ids(tb):
  return {id(row): j for j, row in enumerate(tb.rows)}

def ens(sets):
  print("#n30 ")
  for f in sets:
    tb = Tbl(csv("%s/%s.csv" % (mootdir("classify"), f)))
    y, at = (lambda row: row.labels[kl_at(tb)]), ids(tb)
    def fold(train, test):
      t0 = time.process_time()
      a = aucs(Forest(tb, train, y).votes(test, y))
      print("%s|%.3f |%.3f |%s " % (f, a, time.process_time() - t0,
            " ".join(str(at[id(r)]) for r in test)), flush=True)
    xval(tb.rows, fold)

def reg(sets):
  print("#1:reg n30:reg mean ")
  for f in sets:
    tb = Tbl(csv("%s/%s.csv" % (mootdir("regression"), f)))
    y, at = (lambda row: row[goalcol(tb).at]), ids(tb)
    def fold(train, test):
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
            maes[2], secs[0], secs[1],
            " ".join(str(at[id(r)]) for r in test)), flush=True)
    xval(tb.rows, fold)

def opt(sets):
  print("#1:opt n30:opt rand ")
  for f in sets:
    tb = Tbl(csv("%s/%s.csv" % (mootdir("optimize"), f)))
    for row in tb.rows: row.d2h = tb.disty(row)
    y, at = (lambda row: row.d2h), ids(tb)
    lo = min(r.d2h for r in tb.rows)
    mu = sum(r.d2h for r in tb.rows) / len(tb.rows)
    score = lambda got: 100 * (1 - (got - lo) / (mu - lo + 1e-32))
    best5 = lambda rows: min(r.d2h for r in rows[:5])
    def fold(train, test):
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
            out[2], secs[0], secs[1],
            " ".join(str(at[id(r)]) for r in test)), flush=True)
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
  if   a and a[0] == "--reg": reg(a[1:] or REG)
  elif a and a[0] == "--opt": opt(a[1:])
  else:                       ens(a or ENS)
