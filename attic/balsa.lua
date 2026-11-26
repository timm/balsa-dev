-- balsa.yue: the learner, a Yue port of bald (lisp). Data, bins,
-- trees, forest, auc. Same Park-Miller rand, so same shuffles.
-- The library: no printing, no files beyond csv, no experiments.
-- The rigs (5x2 xval, fold replay, stats) live in balsa_eg.yue.
-- Names say shape: tb col row; v = one cell value, b = a bin #,
-- n = a count, s = a string; fn mk = functions; plurals are lists;
-- sym1, sym2 where only Syms will do; y reads a row's label.
-- Everything counted is a small int LABEL, assigned at load, ONE
-- convention: known values get 1..n (x cells bin #s 1..nbins,
-- class cells 1..nk; each col's legend in its .labels); unknown ?
-- is -1, everywhere. So hot loops walk arrays 1..n and never
-- pairs(), whose order is the version's hash seed, not ours:
-- same seed, same model, on any Lua.
-- (c) 2026 Tim Menzies <timm@ieee.org>, MIT license
local floor, sqrt, max, min, log = math.floor, math.sqrt, math.max, math.min, math.log
local LOG2 = log(2)

local the = {
  seed = 1234567891,
  prudence = 0.01,
  bins = 16,
  trees = 30,
  cap = 200,
  leaf = 2,
  maxd = 30,
  rf = true
}

------------------------------------------------------------- random
local seed = the.seed
local srand
srand = function(n)
  seed = n
end
local rand
rand = function(n)
  if n == nil then
    n = 1
  end
  seed = (16807 * seed) % 2147483647
  return n * seed / 2147483647
end

local shuffle
shuffle = function(rows)
  local u
  do
    local _accum_0 = { }
    local _len_0 = 1
    for _index_0 = 1, #rows do
      local row = rows[_index_0]
      _accum_0[_len_0] = row
      _len_0 = _len_0 + 1
    end
    u = _accum_0
  end
  for i = #u, 2, -1 do
    local j = 1 + floor(rand(i))
    u[i], u[j] = u[j], u[i]
  end
  return u
end

local pick
pick = function(n, m)
  if not (m and m < n) then
    local _accum_0 = { }
    local _len_0 = 1
    for i = 0, n - 1 do
      _accum_0[_len_0] = i
      _len_0 = _len_0 + 1
    end
    return _accum_0
  end
  local got, out = { }, { }
  while #out < m do
    local i = floor(rand(n))
    if not got[i] then
      got[i] = true
      out[#out + 1] = i
    end
  end
  table.sort(out)
  return out
end

---------------------------------------------------------------- csv
local thing
thing = function(cell)
  cell = cell:match("^%s*(.-)%s*$")
  return tonumber(cell) or cell
end

local csv
csv = function(file)
  file = file:gsub("^~", os.getenv("HOME"))
  local rows = { }
  for line in io.lines(file) do
    local txt = line:gsub("\r", ""):match("^%s*(.-)%s*$")
    if txt ~= "" then
      local _accum_0 = { }
      local _len_0 = 1
      for cell in (tostring(txt) .. ","):gmatch("(.-),") do
        _accum_0[_len_0] = thing(cell)
        _len_0 = _len_0 + 1
      end
      rows[#rows + 1] = _accum_0
    end
  end
  return rows
end

--------------------------------------------------------------- data
-- Header: Uppercase = Num, else Sym. Last char: ! class, X skip.
-- "?" cells unknown. A row IS its csv cells (numeric indexes),
-- plus one named field: labels, its label per x column.
local Col
Col = function(txt, at)
  if txt == nil then
    txt = "N"
  end
  if at == nil then
    at = 0
  end
  if txt:match("^%u") then
    return {
      at = at,
      txt = txt,
      isnum = true,
      n = 0,
      mu = 0,
      m2 = 0,
      goal = txt:match("%-$") and 0 or 1
    }
  else
    return {
      at = at,
      txt = txt,
      n = 0,
      has = { }
    }
  end
end

local add
add = function(col, v, inc)
  if inc == nil then
    inc = 1
  end
  if v == "?" then
    return
  end
  col.n = col.n + inc
  if col.isnum then
    local d = v - col.mu
    col.mu = col.mu + (inc * d / col.n)
    col.m2 = max(0, col.m2 + inc * d * (v - col.mu))
  else
    col.has[v] = (col.has[v] or 0) + inc
  end
end

local adds
adds = function(vs, col)
  if col == nil then
    col = Col()
  end
  for _index_0 = 1, #vs do
    local v = vs[_index_0]
    add(col, v)
  end
  return col
end

local div
div = function(col)
  if col.isnum then
    if col.n < 2 then
      return 0
    else
      return sqrt(col.m2 / (col.n - 1))
    end
  end
  local e = 0
  for _, m in pairs(col.has) do
    if m > 0 then
      e = e - (m / col.n * log(m / col.n))
    end
  end
  return e / LOG2
end

-- class counts: an array indexed by label 1..nk
local _anon_func_0 = function(nk)
  local _accum_0 = { }
  local _len_0 = 1
  for _ = 1, nk do
    _accum_0[_len_0] = 0
    _len_0 = _len_0 + 1
  end
  return _accum_0
end
local Ys
Ys = function(nk)
  return {
    nk = nk,
    n = 0,
    has = _anon_func_0(nk)
  }
end

local yadd
yadd = function(ys, v)
  if ys.nk then
    if v > 0 then
      local _obj_0 = ys.has
      _obj_0[v] = _obj_0[v] + 1
      ys.n = ys.n + 1
    end
  else
    return add(ys, v)
  end
end

local yadds
yadds = function(vs, ys)
  for _index_0 = 1, #vs do
    local v = vs[_index_0]
    yadd(ys, v)
  end
  return ys
end

local yclear
yclear = function(ys)
  if ys.nk then
    for i = 1, ys.nk do
      ys.has[i] = 0
    end
    ys.n = 0
  else
    ys.n, ys.mu, ys.m2 = 0, 0, 0
  end
  return ys
end

local _anon_func_1 = function(mu, n, ys1)
  if n < ys1.n then
    return mu
  else
    return ys1.mu
  end
end
local yfold
yfold = function(ys1, ys2, inc, out)
  if inc == nil then
    inc = 1
  end
  if out == nil then
    out = ys1
  end
  if ys1.nk then
    for i = 1, ys1.nk do
      out.has[i] = ys1.has[i] + inc * ys2.has[i]
    end
    out.n = ys1.n + inc * ys2.n
  else
    local m = ys2.n
    local n = ys1.n + inc * m
    if n <= 0 then
      out.n, out.mu, out.m2 = 0, 0, 0
    else
      local mu = ys1.mu + inc * m * (ys2.mu - ys1.mu) / n
      local d = ys2.mu - (_anon_func_1(mu, n, ys1))
      out.m2 = max(0, ys1.m2 + inc * (ys2.m2 + min(n, ys1.n) * m * d * d / max(n, ys1.n)))
      out.mu, out.n = mu, n
    end
  end
  return out
end

local ydiv
ydiv = function(ys)
  if ys.n < 2 then
    return 0
  end
  local e = 0
  local _list_0 = ys.has
  for _index_0 = 1, #_list_0 do
    local c = _list_0[_index_0]
    if c > 0 then
      e = e - (c * log(c))
    end
  end
  return (e / ys.n + log(ys.n)) / LOG2
end

local norm
norm = function(num, v)
  local z = (v - num.mu) / (div(num) + 1e-32)
  z = max(-3, min(3, z))
  return 1 / (1 + math.exp(-1.7 * z))
end

local disty
disty = function(tb, row)
  local d, n = 0, 0
  local _list_0 = tb.cols.y
  for _index_0 = 1, #_list_0 do
    local col = _list_0[_index_0]
    local v = row[col.at]
    if v ~= "?" then
      d = d + ((norm(col, v) - col.goal) ^ 2)
      n = n + 1
    end
  end
  return (d / max(n, 1)) ^ 0.5
end

local mid
mid = function(ys)
  if not ys.nk then
    return ys.mu
  end
  local best, most = nil, -1
  for i = 1, ys.nk do
    if ys.has[i] > most then
      best, most = i, ys.has[i]
    end
  end
  return best
end

local spread
spread = function(ys)
  if ys.nk then
    return ydiv(ys)
  else
    return div(ys)
  end
end

local Cols
Cols = function(names)
  local all
  do
    local _accum_0 = { }
    local _len_0 = 1
    for at, txt in ipairs(names) do
      _accum_0[_len_0] = Col(txt, at)
      _len_0 = _len_0 + 1
    end
    all = _accum_0
  end
  local x, y, klass = { }, { }, nil
  for i = #all, 1, -1 do
    local _continue_0 = false
    repeat
      local col = all[i]
      local e = col.txt:sub(-1)
      if col.txt:match("X$") then
        _continue_0 = true
        break
      end
      if e == "+" or e == "-" or e == "!" then
        y[#y + 1] = col
        if e == "!" then
          klass = col
        end
      else
        x[#x + 1] = col
      end
      _continue_0 = true
    until true
    if not _continue_0 then
      break
    end
  end
  return {
    names = names,
    all = all,
    x = x,
    y = y,
    klass = klass
  }
end

------------- labels: made once at load; trees cut on label #s
local bisect
bisect = function(vs, v, le)
  local lo, hi = 1, #vs + 1
  while lo < hi do
    local mp = floor((lo + hi) / 2)
    local below
    if le then
      below = vs[mp] <= v
    else
      below = vs[mp] < v
    end
    if below then
      lo = mp + 1
    else
      hi = mp
    end
  end
  return lo - 1
end

local mklabels
mklabels = function(col, rows, n)
  if col.isnum then
    local vs
    do
      local _accum_0 = { }
      local _len_0 = 1
      for _index_0 = 1, #rows do
        local row = rows[_index_0]
        if row[col.at] ~= "?" then
          _accum_0[_len_0] = row[col.at]
          _len_0 = _len_0 + 1
        end
      end
      vs = _accum_0
    end
    table.sort(vs)
    local out, b4 = { }, nil
    for j = 0, (#vs > 0 and n - 1 or -1) do
      local v = vs[1 + floor(j * #vs / n)]
      if v ~= b4 then
        out[#out + 1] = v
        b4 = v
      end
    end
    return out
  else
    local vs
    do
      local _accum_0 = { }
      local _len_0 = 1
      for v, m in pairs(col.has) do
        _accum_0[_len_0] = {
          v,
          m
        }
        _len_0 = _len_0 + 1
      end
      vs = _accum_0
    end
    table.sort(vs, function(a, b)
      return a[2] > b[2] or (a[2] == b[2] and tostring(a[1]) < tostring(b[1]))
    end)
    local _accum_0 = { }
    local _len_0 = 1
    for i = 1, min(n - 1, #vs) do
      _accum_0[_len_0] = vs[i][1]
      _len_0 = _len_0 + 1
    end
    return _accum_0
  end
end

local tolabel
tolabel = function(col, v)
  if v == "?" then
    return -1
  end
  if col.isnum then
    return max(1, bisect(col.labels, v, true))
  end
  for i, u in ipairs(col.labels) do
    if u == v then
      return i
    end
  end
  return #col.labels + 1
end

local labelall
labelall = function(tb)
  local kl = tb.cols.klass
  if kl and kl.isnum then
    kl = nil
  end
  local cols
  do
    local _accum_0 = { }
    local _len_0 = 1
    local _list_0 = tb.cols.x
    for _index_0 = 1, #_list_0 do
      local col = _list_0[_index_0]
      _accum_0[_len_0] = col
      _len_0 = _len_0 + 1
    end
    cols = _accum_0
  end
  if kl then
    cols[#cols + 1] = kl
  end
  for _index_0 = 1, #cols do
    local col = cols[_index_0]
    local n
    if col == kl then
      n = 1 / 0
    else
      n = the.bins
    end
    col.labels = mklabels(col, tb.rows, n)
  end
  local _list_0 = tb.rows
  for _index_0 = 1, #_list_0 do
    local row = _list_0[_index_0]
    row.labels = { }
    for _index_1 = 1, #cols do
      local col = cols[_index_1]
      row.labels[col.at] = tolabel(col, row[col.at])
    end
  end
  if kl then
    tb.nk = #kl.labels
  else
    tb.nk = 0
  end
end

local Tbl
Tbl = function(rows)
  local tb = {
    cols = Cols(rows[1]),
    rows = { }
  }
  for i = #rows, 2, -1 do
    local row = rows[i]
    do
      local _obj_0 = tb.rows
      _obj_0[#_obj_0 + 1] = row
    end
    local _list_0 = tb.cols.all
    for _index_0 = 1, #_list_0 do
      local col = _list_0[_index_0]
      add(col, row[col.at])
    end
  end
  labelall(tb)
  return tb
end

--------------------------------------------------------------- tree
local argmost
argmost = function(better)
  local out, b4 = nil, nil
  return function(x, n)
    if x and (b4 == nil or better(n, b4)) then
      out, b4 = x, n
    end
    return out, b4
  end
end

local xpect
xpect = function(a, b)
  return (a.n * spread(a) + b.n * spread(b)) / (a.n + b.n + 1e-32)
end

local cuts
cuts = function(col, rows, ys, mk, leaf, best, pool)
  local nb = #col.labels
  local bins
  do
    local _accum_0 = { }
    local _len_0 = 1
    for b = 1, nb + 1 do
      _accum_0[_len_0] = yclear(pool[b])
      _len_0 = _len_0 + 1
    end
    bins = _accum_0
  end
  for i, row in ipairs(rows) do
    local b = row.labels[col.at]
    if b > 0 then
      yadd(bins[b], ys[i])
    end
  end
  local yes = yclear(pool[nb + 2])
  local no = yclear(pool[nb + 3])
  local rest = yclear(pool[nb + 4])
  for _index_0 = 1, #bins do
    local bin = bins[_index_0]
    yfold(no, bin)
  end
  for b = 1, nb do
    local bin = bins[b]
    local lhs, rhs
    if col.isnum then
      yfold(yes, bin)
      yfold(no, bin, -1)
      lhs, rhs = yes, no
    else
      lhs, rhs = bin, yfold(no, bin, -1, rest)
    end
    if leaf <= min(lhs.n, rhs.n) then
      best({
        col,
        b
      }, xpect(lhs, rhs))
    end
  end
end

local best_cut
best_cut = function(xcols, rows, ys, mk, m, leaf, pool)
  local best = argmost(function(a, b)
    return a < b
  end)
  local _list_0 = pick(#xcols, m)
  for _index_0 = 1, #_list_0 do
    local i = _list_0[_index_0]
    cuts(xcols[i + 1], rows, ys, mk, leaf, best, pool)
  end
  return best()
end

local _anon_func_2 = function(b, col, w)
  if col.isnum then
    return w <= b
  else
    return w == b
  end
end
local yes_
yes_ = function(col, b, row)
  local w = row.labels[col.at]
  return w < 0 or (_anon_func_2(b, col, w))
end

local _anon_func_3 = function(mk, the)
  local _accum_0 = { }
  local _len_0 = 1
  for _ = 1, the.bins + 4 do
    _accum_0[_len_0] = mk()
    _len_0 = _len_0 + 1
  end
  return _accum_0
end
local tree
tree = function(rows, y, xcols, mk, m, leaf, maxd, lvl, pool)
  if lvl == nil then
    lvl = 0
  end
  pool = pool or _anon_func_3(mk, the)
  local ys
  do
    local _accum_0 = { }
    local _len_0 = 1
    for _index_0 = 1, #rows do
      local row = rows[_index_0]
      _accum_0[_len_0] = y(row)
      _len_0 = _len_0 + 1
    end
    ys = _accum_0
  end
  local cut = lvl < maxd and best_cut(xcols, rows, ys, mk, m, leaf, pool)
  local node = {
    ys = yadds(ys, mk()),
    n = #rows
  }
  if cut then
    local col, b = cut[1], cut[2]
    node.v = b
    node.ati, node.isnum = col.at, col.isnum
    local yes
    do
      local _accum_0 = { }
      local _len_0 = 1
      for _index_0 = 1, #rows do
        local row = rows[_index_0]
        if yes_(col, b, row) then
          _accum_0[_len_0] = row
          _len_0 = _len_0 + 1
        end
      end
      yes = _accum_0
    end
    local no
    do
      local _accum_0 = { }
      local _len_0 = 1
      for _index_0 = 1, #rows do
        local row = rows[_index_0]
        if not yes_(col, b, row) then
          _accum_0[_len_0] = row
          _len_0 = _len_0 + 1
        end
      end
      no = _accum_0
    end
    node.yes = tree(yes, y, xcols, mk, m, leaf, maxd, lvl + 1, pool)
    node.no = tree(no, y, xcols, mk, m, leaf, maxd, lvl + 1, pool)
  end
  return node
end

local _anon_func_4 = function(node, w)
  if node.isnum then
    return w <= node.v
  else
    return w == node.v
  end
end
local leaf_of
leaf_of = function(node, row)
  while node.ati do
    local w = row.labels[node.ati]
    local ok = w < 0 or (_anon_func_4(node, w))
    node = ok and node.yes or node.no
  end
  return node
end

------------------------------------------------------------- forest
local capped
capped = function(rows, y, n, nk)
  if not nk then
    local u = shuffle(rows)
    local _accum_0 = { }
    local _len_0 = 1
    for i = 1, min(n, #u) do
      _accum_0[_len_0] = u[i]
      _len_0 = _len_0 + 1
    end
    return _accum_0
  end
  local h, out = { }, { }
  local _list_0 = shuffle(rows)
  for _index_0 = 1, #_list_0 do
    local row = _list_0[_index_0]
    local v = y(row)
    h[v] = (h[v] or 0) + 1
    if h[v] <= n then
      out[#out + 1] = row
    end
  end
  return out
end

local forest
forest = function(tb, rows, y, mk)
  local xcols = tb.cols.x
  mk = mk or function()
    return Ys(tb.nk)
  end
  local nk = mk().nk
  local m = nil
  if the.rf then
    if nk then
      m = math.ceil(sqrt(#xcols))
    else
      m = math.ceil(#xcols / 3)
    end
  end
  --                                       but p/3 for regression
  if the.cap then
    rows = capped(rows, y, the.cap, nk)
  end
  local half
  half = function()
    local u = shuffle(rows)
    local _accum_0 = { }
    local _len_0 = 1
    for i = 1, floor(#rows / 2) do
      _accum_0[_len_0] = u[i]
      _len_0 = _len_0 + 1
    end
    return _accum_0
  end
  local out = { }
  for _ = 1, the.trees do
    out[#out + 1] = tree(the.trees == 1 and rows or half(), y, xcols, mk, m, the.leaf, the.maxd)
  end
  return out
end

local _anon_func_5 = function(nk)
  local _accum_0 = { }
  local _len_0 = 1
  for _ = 1, nk do
    _accum_0[_len_0] = 0
    _len_0 = _len_0 + 1
  end
  return _accum_0
end
local vote
vote = function(trees, row)
  local nk = trees[1].ys.nk
  local h, w = _anon_func_5(nk), #trees
  for _index_0 = 1, #trees do
    local node = trees[_index_0]
    local ys = leaf_of(node, row).ys
    for i = 1, nk do
      h[i] = h[i] + (ys.has[i] / ys.n / w)
    end
  end
  return h
end

local votes
votes = function(trees, rows, y)
  local _accum_0 = { }
  local _len_0 = 1
  for _index_0 = 1, #rows do
    local row = rows[_index_0]
    _accum_0[_len_0] = {
      y(row),
      vote(trees, row)
    }
    _len_0 = _len_0 + 1
  end
  return _accum_0
end

local predict
predict = function(trees, row)
  local s = 0
  for _index_0 = 1, #trees do
    local node = trees[_index_0]
    s = s + mid(leaf_of(node, row).ys)
  end
  return s / #trees
end

---------------------------------------------------------------- auc
local auc
auc = function(pos, neg)
  if #pos == 0 or #neg == 0 then
    return nil
  end
  table.sort(neg)
  local s = 0
  for _index_0 = 1, #pos do
    local p = pos[_index_0]
    s = s + ((bisect(neg, p) + bisect(neg, p, true)) / 2)
  end
  return s / (#pos * #neg)
end

local aucs
aucs = function(wps)
  local n, sum = 0, 0
  for c = 1, #wps[1][2] do
    local pos
    do
      local _accum_0 = { }
      local _len_0 = 1
      for _index_0 = 1, #wps do
        local wp = wps[_index_0]
        if wp[1] == c then
          _accum_0[_len_0] = wp[2][c]
          _len_0 = _len_0 + 1
        end
      end
      pos = _accum_0
    end
    local neg
    do
      local _accum_0 = { }
      local _len_0 = 1
      for _index_0 = 1, #wps do
        local wp = wps[_index_0]
        if wp[1] ~= c then
          _accum_0[_len_0] = wp[2][c]
          _len_0 = _len_0 + 1
        end
      end
      neg = _accum_0
    end
    local a = auc(pos, neg)
    if a then
      n = n + (#pos)
      sum = sum + (#pos * a)
    end
  end
  return 100 * sum / max(n, 1)
end

-- the module: what balsa_eg.yue (and any other rig) may touch
return {
  the = the,
  srand = srand,
  rand = rand,
  shuffle = shuffle,
  bisect = bisect,
  csv = csv,
  Tbl = Tbl,
  Col = Col,
  adds = adds,
  div = div,
  mid = mid,
  disty = disty,
  tree = tree,
  leaf_of = leaf_of,
  forest = forest,
  votes = votes,
  predict = predict,
  auc = auc,
  aucs = aucs
}
