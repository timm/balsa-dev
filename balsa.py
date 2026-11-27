# vim: set ts=2 sw=2 sts=2 et cc=75:
"""
balsa: lightweight ensemble trees for classification, regression
(c) 2026 Tim Menzies <timm@ieee.org>, MIT license

Options:
  seed=1234567891   random number seed (Park-Miller)
  prudence=0.01     mean gaps under this are never 'different'
  bins=16           labels per numeric column (syms: commonest 15)
  trees=30          trees per forest
  cap=200           rows per class (regression: rows, total)
  leaf=2            no cut may leave fewer rows than this
  maxd=30           max tree depth
  rf=True           cols per cut: sqrt(|x|); regression |x|/3
"""
import re
from bisect import bisect_left, bisect_right
from math import ceil, exp, floor, inf, log, nan, sqrt
from os.path import expanduser

isa = isinstance

# root all things -------------------------------------------------------
def say(x):
  if   isa(x, float):          x = ("%.2f" % x).rstrip("0").rstrip(".")
  elif isa(x, (list, tuple)):  x = "[%s]" % ", ".join(map(say, x))
  elif hasattr(x,"__slots__"): x = {k:getattr(x,k) for k in x.__slots__}
  if   isa(x, dict):  x = "{%s}"%" ".join(f":{k} {say(x[k])}" for k in x)
  return str(x)

class Box:
  def __init__(i, **d) : i.__dict__.update(**d)
  def __repr__(i)      : type(i).__name__ + say(vars(i))

# config ----------------------------------------------------------------
def thing(s):
  for f in (int, float):
    try: return f(s)
    except ValueError: pass
  return {"True": True, "False": False}.get(s, s.strip())

the = Box(**{k: thing(v) for k,v in re.findall(r"(\w+)=(\S+)",__doc__)})

# csv ------------------------------------------------------------------
# CSV rows. Headers start uppercase for numbers. Suffixes !, +, -, and X
# mark the class, goals to maximize or minimize, and columns to skip.
class Row(list):
  __slots__ = ("labels", "d2h")

def csv(file):
  with open(expanduser(file), encoding="utf-8") as f:
    return [Row(thing(v) for v in line.split(","))
            for s in f if (line := s.replace("\r", "").strip())]

# Column ----------------------------------------------------------------
# Column and target summaries. Known labels are 0..n-1; "?" is -1,
# so "unknown" is always label < 0.
class Column(Box):
  def __init__(i, txt, at): i.txt, i.at, i.n = txt, at, 0

  def add(i, v, inc=1):
    if v != "?":
      i.n += inc
      i.add_known(v, inc)

  def cuts(i, rows, targets, mk, leaf):
    bins = [mk() for _ in range(len(i.labels) + 1)]
    for row, y in zip(rows, targets):
      b = row.labels[i.at]
      if b >= 0: bins[b].add(y)
    prefix, total, rest = mk(), mk(), mk()
    for bin in bins: yfold(total, bin)
    for b, bin in enumerate(bins[:-1]):
      left, right = i.split_bin(bin, prefix, total, rest)
      if min(left.n, right.n) >= leaf:
        yield (i, b), xpect(left, right)

def Col(txt="N", at=0):
  return (Num if "A" <= txt[:1] <= "Z" else Sym)(txt, at)

# Sym ----------------------------------------------------------------
class Sym(Column):
  def __init__(i, txt, at):
    super().__init__(txt, at)
    i.has = {}

  def add_known(i, v, inc): i.has[v] = i.has.get(v, 0) + inc

  def div(i):
    return -sum((p := n/i.n)*log(p) for n in i.has.values() if n>0)/log(2)

  def mid(i): return None

  def labels_for(i, n, rows=None):
    def key(v):
      txt = format(v, ".14g") if isa(v, float) else str(v)
      return -i.has[v], txt
    ranked = sorted(i.has, key=key)
    return ranked[:max(0, min(len(ranked), n - 1))]

  def label(i, v):
    try: return i.labels.index(v)
    except ValueError: return len(i.labels)   # the rare pool

  def class_labels(i): return i.labels_for(inf)

  def split_bin(i, bin, prefix, total, rest):
    return bin, yfold(total, bin, -1, rest)

  def matches(i, label, cut): return label == cut

# Num ----------------------------------------------------------------
def bisect(t, v, le=False):
  return (bisect_right if le else bisect_left)(t, v)

class Num(Column):
  def __init__(i, txt="N", at=0):
    super().__init__(txt, at)
    i.mu, i.m2, i.goal = 0, 0, int(not txt.endswith("-"))

  def add_known(i, v, inc):
    d = v - i.mu
    i.mu += inc * d / i.n
    i.m2 = max(0, i.m2 + inc * d * (v - i.mu))

  def div(i): return 0 if i.n < 2 else sqrt(i.m2 / (i.n - 1))

  def mid(i): return i.mu

  def fold(i, b, inc, u):
    n = i.n + inc * b.n
    if n <= 0:
      u.n, u.mu, u.m2 = 0, 0, 0
      return u
    mu = i.mu + inc * b.n * (b.mu - i.mu) / n
    gap = b.mu - (mu if n < i.n else i.mu)
    between = min(n, i.n) * b.n * gap * gap / max(n, i.n)
    u.m2 = max(0, i.m2 + inc * (b.m2 + between))
    u.mu, u.n = mu, n
    return u

  def norm(i, v):
    z = (v - i.mu) / (i.div() + 1e-32)
    return 1 / (1 + exp(-1.7 * max(-3, min(3, z))))

  def labels_for(i, n, rows):
    values = sorted(row[i.at] for row in rows if row[i.at] != "?")
    if not values: return []
    m = len(values)
    return list(dict.fromkeys(values[j * m // n] for j in range(n)))

  def label(i, v): return max(0, bisect_right(i.labels, v) - 1)

  def split_bin(i, bin, prefix, total, rest):
    return yfold(prefix, bin), yfold(total, bin, -1)

  def matches(i, label, cut): return label <= cut

# Targets -------------------------------------------------------------
class Ys(Box):
  def __init__(i, nk): i.nk, i.n, i.has = nk, 0, [0] * nk

  def add(i, v):
    if v >= 0:
      i.has[v] += 1
      i.n += 1

  def div(i):
    if i.n < 2: return 0
    e = sum(-c * log(c) for c in i.has if c > 0)
    return (e / i.n + log(i.n)) / log(2)

  def mid(i): return i.has.index(max(i.has)) if i.has else None

  def fold(i, b, inc, u):
    for j, c in enumerate(b.has):
      u.has[j] = i.has[j] + inc * c
    u.n = i.n + inc * b.n
    return u

# xxxxx y -------------------------------------------------------------
def adds(vs, col=None):
  col = Num() if col is None else col
  for v in vs: col.add(v)
  return col

def yfold(ys, other, inc=1, u=None):
  return ys.fold(other, inc, ys if u is None else u)

def tolabel(col, v): return -1 if v == "?" else col.label(v)

# Tables -------------------------------------------------------------
class Tbl(Box):
  def __init__(i, rows):
    i.cols = Cols(rows[0])
    i.rows = [i.cols.add(row) for row in rows[1:]]
    i.labelall()

  def labelall(i):
    cols, klass, i.nk = i.cols.x[:], i.cols.klass, 0
    for col in cols: col.labels = col.labels_for(the.bins, i.rows)
    if isa(klass, Sym):
      klass.labels = klass.class_labels()
      cols.append(klass)
      i.nk = len(klass.labels)
    for row in i.rows:
      row.labels = [None] * len(row)
      for col in cols: row.labels[col.at] = tolabel(col, row[col.at])

  def disty(i, row):
    d, n = 0, 0
    for col in i.cols.y:
      v = row[col.at]
      if v != "?":
        d += (col.norm(v) - col.goal) ** 2
        n += 1
    return sqrt(d / max(n, 1))

class Cols(Box):
  def __init__(i, names):
    i.names, i.x, i.y, i.klass = names, [], [], None
    i.all = [Col(txt, j) for j, txt in enumerate(names)]
    for col in i.all:
      suffix = col.txt[-1:]
      if suffix == "X": continue
      (i.y if suffix in ("+", "-", "!") else i.x).append(col)
      if suffix == "!": i.klass = col

  def add(i, row):
    row = row if isa(row, Row) else Row(row)
    for col in i.all: col.add(row[col.at])
    return row

# Tree -------------------------------------------------------------
# Split search and trees. Numeric cuts use a threshold, symbolic cuts
# use equality, and unknowns go left.
def xpect(left, right):
  return ((left.n * left.div() + right.n * right.div())
          / (left.n + right.n + 1e-32))

def best_cut(xcols, rows, targets, mk, m, leaf):
  cuts = (c for col in pick(xcols, m)
          for c in col.cuts(rows, targets, mk, leaf))
  return min(cuts, key=lambda c: c[1], default=(None, None))[0]

class Tree(Box):
  def __init__(i, rows, y, xcols, mk, m, leaf, maxd, lvl=0):
    targets = [y(row) for row in rows]
    i.ys, i.n, i.yes = adds(targets, mk()), len(rows), None
    if lvl < maxd:
      if cut := best_cut(xcols, rows, targets, mk, m, leaf):
        i.col, i.v = cut
        yes, no = [], []
        for row in rows:
          (yes if i.goes_left(row) else no).append(row)
        i.yes = Tree(yes, y, xcols, mk, m, leaf, maxd, lvl + 1)
        i.no  = Tree(no,  y, xcols, mk, m, leaf, maxd, lvl + 1)

  def goes_left(i, row):
    label = row.labels[i.col.at]
    return label < 0 or i.col.matches(label, i.v)

  def leaf_of(i, row):
    while i.yes: i = i.yes if i.goes_left(row) else i.no
    return i

# Sampling -------------------------------------------------------------
_seed = the.seed

def srand(n):
  global _seed
  _seed = n

def rand(n=1):
  global _seed
  _seed = 16807 * _seed % 2147483647
  return 1 + floor(n * _seed / 2147483647)

def shuffle(t):
  u = t[:1]
  for j in range(1, len(t)):
    k = rand(j + 1) - 1
    u.append(u[k] if k < j else t[j])
    u[k] = t[j]
  return u

def pick(t, n=None):
  n = n or len(t)
  return shuffle(t)[:ceil(n)] if n < len(t) else t[:]

def capped(rows, y, n, nk=None):
  shuffled = shuffle(rows)
  if nk is None or nk is False: return shuffled[:max(0, floor(n))]
  counts, kept = {}, []
  for row in shuffled:
    label = y(row)
    counts[label] = counts.get(label, 0) + 1
    if counts[label] <= n: kept.append(row)
  return kept

# Forests -------------------------------------------------------------
class Forest(list):
  def __init__(i, tb, rows, y, mk=None):
    if mk is None: mk = lambda: Ys(tb.nk)
    nk = getattr(mk(), "nk", None)
    xcols, m = tb.cols.x, None
    if the.rf:
      m = ceil(sqrt(len(xcols)) if nk is not None else len(xcols) / 3)
    if the.cap is not False and the.cap is not None:
      rows = capped(rows, y, the.cap, nk)
    for _ in range(the.trees):
      sample = rows if the.trees == 1 else pick(rows, len(rows) // 2)
      i.append(Tree(sample, y, xcols, mk, m, the.leaf, the.maxd))

  def vote(i, row):
    nk, w = i[0].ys.nk, len(i)
    h = [0] * nk
    for node in i:
      ys = node.leaf_of(row).ys
      for j, c in enumerate(ys.has):
        h[j] += c / ys.n / w if ys.n else nan
    return h

  def votes(i, rows, y): return [(y(row), i.vote(row)) for row in rows]

  def predict(i, row):
    if not i: return nan
    return sum(node.leaf_of(row).ys.mid() for node in i) / len(i)

# Scoring  -------------------------------------------------------------
def auc(pos, neg):
  if not pos or not neg: return None
  neg.sort()
  s = sum((bisect_left(neg, p) + bisect_right(neg, p)) / 2 for p in pos)
  return s / (len(pos) * len(neg))

def aucs(wps):
  n, weighted = 0, 0
  for j in range(len(wps[0][1])):
    pos, neg = [], []
    for y, scores in wps:
      (pos if y == j else neg).append(scores[j])
    a = auc(pos, neg)
    if a is not None: n, weighted = n + len(pos), weighted + len(pos) * a
  return 100 * weighted / max(n, 1)
