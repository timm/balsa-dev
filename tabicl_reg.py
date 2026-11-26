#!/usr/bin/env python3 -B
"""tabicl_reg: add TabICL regression to balsa_eg --reg's table, on the
same 5x2 folds as balsa and LightGBM.

Reads a .dat from `balsa_eg --reg` (data|mae1 mae30 maeMean|secs|test
row ids), refits TabICLRegressor per fold on the complement of the
recorded test rows, and reports RELATIVE mae: mae / guess-the-mean mae,
so 1.0 is no better than predicting the mean and smaller is better.
Column selection, goal selection and symbol coding are lgbm_reg.load's,
so all three rigs score the same y from the same x.

ROW CAP.  TabICL is an in-context learner: every training row goes into
the transformer's context, so cost and memory climb faster than linearly
in rows.  Measured on this corpus, one fold costs about
3.3e-4 * (rows/2)^1.4 seconds and 0.5 + 2.1*(rows/2)/5500 GB.  At
ROWCAP=20000 the whole corpus runs in about two hours under 9GB; with no
cap, `systems/javagc` alone would want 32GB for a single fold.  Sets over
the cap are skipped and named in the output, the way the classification
figure names the sets TabICL could not finish.

MEMORY.  Thread env vars are pinned to 1 (TabICL starts its own joblib
pool per worker), batch_size=1 keeps one ensemble member in flight, big
sets get fewer workers, and roomy() blocks while free RAM is under
MEM_FLOOR_GB.  This is the configuration that stopped the run from
wedging a 16GB machine.

CRASHES.  Every fold is appended to the jsonl as it finishes, so a kill
costs the folds in flight, not the run.  Re-running skips what the jsonl
already holds.  The table is printed from the jsonl.

usage: tabicl_reg.py runs/reg56.dat runs/reg56-tabicl.jsonl > runs/reg56-tabicl.txt
       tabicl_reg.py ... --report > ...        (print from jsonl, run nothing)
"""
import os, sys, json, time, signal, warnings, multiprocessing as mp

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LOKY_MAX_CPU_COUNT"):
  os.environ.setdefault(v, "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lgbm_reg import load

Small   = int(os.getenv("SMALL_WORKERS", "5"))    # sets under BigRows
Big     = int(os.getenv("BIG_WORKERS", "2"))      # sets at or over BigRows
BigRows = int(os.getenv("BIG_ROWS", "3000"))
RowCap  = int(os.getenv("ROWCAP", "20000"))
Batch   = int(os.getenv("BATCH", "1"))
Device  = os.getenv("DEVICE", "cpu")
Floor   = float(os.getenv("MEM_FLOOR_GB", "4"))
Timeout = int(os.getenv("TABICL_TIMEOUT", "1800"))
Data    = {}


def data(f):
  "Load once per worker."
  if f not in Data: Data[f] = load(f)
  return Data[f]


def roomy():
  "Block while free RAM is under the floor.  A spike here froze the box."
  try:
    import psutil
  except ImportError:
    return
  for _ in range(600):
    if psutil.virtual_memory().available / 2 ** 30 >= Floor: return
    time.sleep(1)


class Late(Exception): pass


def one(job):
  "Score one fold.  Returns the jsonl record, or an error record."
  f, i, base, test = job
  roomy()
  try:
    from tabicl import TabICLRegressor
    X, y, _ = data(f)
    ok    = [j for j in range(len(y)) if not np.isnan(y[j])]
    tset  = set(test)
    train = [j for j in ok if j not in tset]
    te    = [j for j in ok if j in tset]
    if not train or not te: return dict(data=f, fold=i, err="empty split")
    # TabICL takes no NaN; impute each x column at its training median,
    # which is what a split on that column would have done anyway.
    Xtr, Xte = X[train].copy(), X[te].copy()
    for c in range(Xtr.shape[1]):
      m = np.nanmedian(Xtr[:, c])
      if np.isnan(m): m = 0.0
      Xtr[np.isnan(Xtr[:, c]), c] = m
      Xte[np.isnan(Xte[:, c]), c] = m
    def bell(*_): raise Late()
    signal.signal(signal.SIGALRM, bell); signal.alarm(Timeout)
    t0 = time.time()
    try:
      m = TabICLRegressor(batch_size=Batch, device=Device)
      m.fit(Xtr, y[train])
      got = m.predict(Xte)
    finally:
      signal.alarm(0)
    mae = float(np.mean(np.abs(y[te] - got)))
    return dict(data=f, fold=i, rel=mae / base, mae=mae,
                secs=time.time() - t0)
  except Late:
    return dict(data=f, fold=i, err="timeout %ds" % Timeout)
  except Exception as e:
    return dict(data=f, fold=i, err="%s: %s" % (type(e).__name__, e))


def folds(dat):
  "set -> [(fold index, guess-the-mean mae, test row ids)...]"
  out = {}
  for line in open(dat).readlines()[1:]:
    if "|" not in line: continue
    f, a, _s, t = line.strip().split("|")
    out.setdefault(f, []).append(
        (len(out.get(f, [])), float(a.split()[2]), [int(v) for v in t.split()]))
  return out


def rows(f):
  root = os.getenv("MOOTREG", "~/gits/moot/regression")
  p = os.path.expanduser("%s/%s.csv" % (root, f))
  return sum(1 for l in open(p) if l.strip()) - 1


def main():
  dat, jsonl = sys.argv[1], sys.argv[2]
  report = "--report" in sys.argv
  F = folds(dat)
  N = {f: rows(f) for f in F}

  have = set()
  if os.path.exists(jsonl):
    for line in open(jsonl):
      r = json.loads(line)
      if "rel" in r: have.add((r["data"], r["fold"]))

  if not report:
    # smallest first, one set at a time, so a kill loses the cheap work
    for f in sorted(F, key=lambda f: N[f]):
      if N[f] > RowCap:
        print("#skip %s (%d rows > ROWCAP %d)" % (f, N[f], RowCap),
              file=sys.stderr, flush=True)
        continue
      todo = [(f, i, base, test) for i, base, test in F[f]
              if (f, i) not in have]
      if not todo: continue
      w = Big if N[f] >= BigRows else Small
      t0 = time.time()
      with open(jsonl, "a") as out, mp.Pool(min(w, len(todo))) as pool:
        for r in pool.imap_unordered(one, todo):
          out.write(json.dumps(r) + "\n"); out.flush()
          if "err" in r:
            print("#fail %s fold %d: %s" % (r["data"], r["fold"], r["err"]),
                  file=sys.stderr, flush=True)
      print("#done %-32s %6d rows %7.1fs" % (f, N[f], time.time() - t0),
            file=sys.stderr, flush=True)

  # the table, printed from the jsonl
  got = {}
  secs = 0.0
  for line in open(jsonl):
    r = json.loads(line)
    if "rel" not in r: continue
    got.setdefault(r["data"], {})[r["fold"]] = r["rel"]
    secs += r.get("secs", 0.0)
  print("#%-17s%s" % ("data", "tabicl"))
  mus = []
  for f in sorted(got):
    v = [got[f][k] for k in sorted(got[f])]
    if len(v) < len(F[f]):
      print("#partial %s: %d of %d folds" % (f, len(v), len(F[f])),
            file=sys.stderr)
      continue
    mu = sum(v) / len(v); mus.append(mu)
    print("%-18s%5.2f" % (f, mu))
  skip = sorted([f for f in F if N[f] > RowCap], key=lambda f: N[f])
  if mus:
    print("\n#mean rel-mae     %5.2f  (%d sets)" % (sum(mus) / len(mus),
                                                    len(mus)))
  print("#secs (all data)  %.1f" % secs)
  print("#skipped %d sets over ROWCAP=%d: %s" % (len(skip), RowCap,
                                                 " ".join(skip)))
  for f in sorted(got):
    v = [got[f][k] for k in sorted(got[f])]
    if len(v) == len(F[f]):
      print("%%folds %s tabicl %s" % (f, " ".join("%.3f" % x for x in v)))


if __name__ == "__main__":
  main()
