-- vim: set ts=2 sw=2 sts=2 et cc=75:
-- Balsa learner: CSV data, bins, decision trees, forests, and AUC.
-- (c) 2026 Tim Menzies <timm@ieee.org>, MIT license
-- Keep module functions private while reading standard Lua globals.
local globals = _ENV
local _ENV = setmetatable({}, {__index = globals})

local floor, sqrt, max, min, log = math.floor, math.sqrt, math.max, math.min, math.log
local LOG2 = log(2)

the = {seed=1234567891, prudence=0.01, bins=16, trees=30,
       cap=200, leaf=2, maxd=30, rf=true}

-- Random -------------------------------------------------------------------
local seed = the.seed
function srand(n)
  seed = n end
function rand(n)
  n = (n == nil) and 1 or n
  seed = (16807 * seed) % 2147483647
  return n * seed / 2147483647 end

function shuffle(rows)
  local u,j = {}
  for _, row in ipairs(rows) do
    u[#u + 1] = row end
  for i = #u, 2, -1 do
    j = 1 + floor(rand(i))
    u[i], u[j] = u[j], u[i] end
  return u end

function pick(n, m)
  local got,out,i
  out = {}
  if not (m and m < n) then
    for j = 0, n - 1 do
      out[#out + 1] = j end
    return out end
  got = {}
  while #out < m do
    i = floor(rand(n))
    if not got[i] then
      got[i] = true
      out[#out + 1] = i end end
  table.sort(out)
  return out end

-- CSV ----------------------------------------------------------------------
function thing(cell)
  cell = cell:match("^%s*(.-)%s*$")
  return tonumber(cell) or cell end

function csv(file)
  local rows,txt,row = {}
  file = file:gsub("^~", os.getenv("HOME"))
  for line in io.lines(file) do
    txt = line:gsub("\r", ""):match("^%s*(.-)%s*$")
    if txt ~= "" then
      row = {}
      for cell in (txt .. ","):gmatch("(.-),") do
        row[#row + 1] = thing(cell) end
      rows[#rows + 1] = row end end
  return rows end

-- Data ---------------------------------------------------------------------
-- Header: Uppercase = Num, else Sym. Last char: ! class, X skip.
-- "?" cells unknown. A row IS its csv cells (numeric indexes),
-- plus one named field: labels, its label per x column.
local Num, Sym, Counts = {}, {}, {}

function new(mt,t)
  mt.__index = mt
  return setmetatable(t, mt) end

-- Plain summaries from callers can use the same methods.
local function kind(t)
  return t.isnum and Num or Sym end

function Col(txt, at)
  txt = (txt == nil) and "N" or txt
  at = (at == nil) and 0 or at
  if txt:match("^%u") then
    return new(Num, {at=at, txt=txt, n=0, isnum=true, mu=0, m2=0,
                     goal=txt:match("%-$") and 0 or 1}) end
  return new(Sym, {at=at, txt=txt, n=0, has={}}) end

function add(col, v, inc)
  inc = (inc == nil) and 1 or inc
  if v == "?" then return end
  col.n = col.n + inc
  return (col.add_known or kind(col).add_known)(col, v, inc) end
function Num:add_known(v, inc)
  local d = v - self.mu
  self.mu = self.mu + inc * d / self.n
  self.m2 = max(0, self.m2 + inc * d * (v - self.mu)) end
function Sym:add_known(v, inc)
  self.has[v] = (self.has[v] or 0) + inc end

function adds(vs, col)
  col = (col == nil) and Col() or col
  for i = 1, #vs do
    add(col, vs[i]) end
  return col end

function div(col)
  return (col.div or kind(col).div)(col) end
function Num:div()
  return self.n < 2 and 0 or sqrt(self.m2 / (self.n - 1)) end
function Sym:div()
  local e = 0
  for _, m in pairs(self.has) do
    if m > 0 then
      e = e - (m / self.n * log(m / self.n)) end end
  return e / LOG2 end

-- class counts: an array indexed by label 1..nk
function zeros(nk)
  local out = {}
  for i = 1, nk do
    out[i] = 0 end
  return out end
function Ys(nk)
  return new(Counts, {nk=nk, n=0, has=zeros(nk)}) end

function yadd(ys, v)
  return (ys.yadd or (ys.nk and Counts.yadd or add))(ys, v) end
function Counts:yadd(v)
  if v > 0 then
    self.has[v] = self.has[v] + 1
    self.n = self.n + 1 end end

function yadds(vs, ys)
  for i = 1, #vs do
    yadd(ys, vs[i]) end
  return ys end

function yfold(a, b, inc, out)
  inc = (inc == nil) and 1 or inc
  out = (out == nil) and a or out
  return (a.fold or (a.nk and Counts.fold or Num.fold))(a, b, inc, out) end
function Counts:fold(b, inc, out)
  for i = 1, self.nk do
    out.has[i] = self.has[i] + inc * b.has[i] end
  out.n = self.n + inc * b.n
  return out end
function Num:fold(b, inc, out)
  local n,mu,gap,between = self.n + inc * b.n
  if n <= 0 then
    out.n, out.mu, out.m2 = 0, 0, 0
    return out end
  mu = self.mu + inc * b.n * (b.mu - self.mu) / n
  -- The gap uses the mean of the smaller summary.
  gap = b.mu - (n < self.n and mu or self.mu)
  between = min(n, self.n) * b.n * gap * gap / max(n, self.n)
  out.m2 = max(0, self.m2 + inc * (b.m2 + between))
  out.mu, out.n = mu, n
  return out end

function ydiv(ys)
  return Counts.spread(ys) end
function spread(ys)
  return (ys.spread or (ys.nk and Counts.spread or div))(ys) end
function Counts:spread()
  local e = 0
  if self.n < 2 then return 0 end
  for _, c in ipairs(self.has) do
    if c > 0 then
      e = e - (c * log(c)) end end
  return (e / self.n + log(self.n)) / LOG2 end

function norm(num, v)
  local z = (v - num.mu) / (div(num) + 1e-32)
  z = max(-3, min(3, z))
  return 1 / (1 + math.exp(-1.7 * z)) end

function disty(tb, row)
  local d,n,v = 0, 0
  for _, col in ipairs(tb.cols.y) do
    v = row[col.at]
    if v ~= "?" then
      d = d + ((norm(col, v) - col.goal) ^ 2)
      n = n + 1 end end
  return (d / max(n, 1)) ^ 0.5 end

function mid(ys)
  return (ys.mid or (ys.nk and Counts.mid or Num.mid))(ys) end
function Num:mid()
  return self.mu end
function Counts:mid()
  local best,most = nil, -1
  for i = 1, self.nk do
    if self.has[i] > most then
      best, most = i, self.has[i] end end
  return best end

function Cols(names)
  local all,x,y,klass,col,e = {}, {}, {}, nil
  for at, txt in ipairs(names) do
    all[at] = Col(txt, at) end
  for i = #all, 1, -1 do
    col = all[i]
    e = col.txt:sub(-1)
    if e ~= "X" then
      if e == "+" or e == "-" or e == "!" then
        y[#y + 1] = col
        if e == "!" then
          klass = col end
      else
        x[#x + 1] = col end end end
  return {names=names, all=all, x=x, y=y, klass=klass} end

-- Labels -------------------------------------------------------------------
-- Known values use positive integers, and "?" is -1. Each column's
-- labels table records the numeric bin boundaries or symbolic values.
function bisect(vs, v, le)
  local lo,hi,mp,below = 1, #vs + 1
  while lo < hi do
    mp = floor((lo + hi) / 2)
    below = le and vs[mp] <= v or (not le and vs[mp] < v)
    if below then
      lo = mp + 1
    else
      hi = mp end end
  return lo - 1 end

function mklabels(col, rows, n)
  return (col.labels_for or kind(col).labels_for)(col, rows, n) end
function Num:labels_for(rows, n)
  local labels,values,v,last = {}, {}
  for _, row in ipairs(rows) do
    v = row[self.at]
    if v ~= "?" then values[#values + 1] = v end end
  if #values == 0 then return labels end
  table.sort(values)
  for j = 0, n - 1 do
    v = values[1 + floor(j * #values / n)]
    if v ~= last then
      labels[#labels + 1] = v
      last = v end end
  return labels end
function Sym:labels_for(_, n)
  local labels,ranked = {}, {}
  for value, count in pairs(self.has) do
    ranked[#ranked + 1] = {value, count} end
  table.sort(ranked, function(a, b)
    if a[2] == b[2] then
      return tostring(a[1]) < tostring(b[1]) end
    return a[2] > b[2] end)
  for i = 1, min(n - 1, #ranked) do
    labels[i] = ranked[i][1] end
  return labels end

function tolabel(col, v)
  if v == "?" then
    return -1 end
  if col.isnum then
    return max(1, bisect(col.labels, v, true)) end
  for i, u in ipairs(col.labels) do
    if u == v then
      return i end end
  return #col.labels + 1 end

function labelall(tb)
  local klass,cols = tb.cols.klass, {}
  for i, col in ipairs(tb.cols.x) do
    col.labels = mklabels(col, tb.rows, the.bins)
    cols[i] = col
  end
  tb.nk = 0
  if klass and not klass.isnum then
    -- Keep every class label.
    klass.labels = mklabels(klass, tb.rows, math.huge)
    cols[#cols + 1] = klass
    tb.nk = #klass.labels
  end
  for _, row in ipairs(tb.rows) do
    row.labels = {}
    for _, col in ipairs(cols) do
      row.labels[col.at] = tolabel(col, row[col.at]) end end end

function Tbl(rows)
  local tb,row = {cols = Cols(rows[1]), rows = {}}
  for i = #rows, 2, -1 do
    row = rows[i]
    tb.rows[#tb.rows + 1] = row
    for _, col in ipairs(tb.cols.all) do
      add(col, row[col.at]) end end
  labelall(tb)
  return tb end

-- Trees --------------------------------------------------------------------
function argmost(better)
  local out,b4 = nil, nil
  return function(x, n)
    if x and (b4 == nil or better(n, b4)) then
      out, b4 = x, n end
    return out, b4 end end

function xpect(a, b)
  return (a.n * spread(a) + b.n * spread(b)) / (a.n + b.n + 1e-32) end

function cuts(col, rows, ys, mk, leaf, best)
  local nb,bins,label,prefix,total,rest,left,right = #col.labels, {}
  for b = 1, nb + 1 do
    bins[b] = mk() end
  for i, row in ipairs(rows) do
    label = row.labels[col.at]
    if label > 0 then
      yadd(bins[label], ys[i]) end
  end
  prefix, total, rest = mk(), mk(), mk()
  for _, bin in ipairs(bins) do
    yfold(total, bin)
  end
  for b = 1, nb do
    if col.isnum then
      -- Move this bin from the right side to the growing prefix.
      left = yfold(prefix, bins[b])
      right = yfold(total, bins[b], -1)
    else
      left = bins[b]
      right = yfold(total, left, -1, rest) end
    if leaf <= min(left.n, right.n) then
      best({col, b}, xpect(left, right)) end end end

function best_cut(xcols, rows, ys, mk, m, leaf)
  local best = argmost(function(a, b) return a < b end)
  for _, i in ipairs(pick(#xcols, m)) do
    cuts(xcols[i + 1], rows, ys, mk, leaf, best) end
  return best() end

function goes_left(label, cut, isnum)
  -- Unknowns follow the left branch during training and prediction.
  return label < 0 or label == cut or (isnum and label < cut) end

function tree(rows, y, xcols, mk, m, leaf, maxd, lvl)
  local ys,cut,node,col,yes,no,branch = {}
  lvl = (lvl == nil) and 0 or lvl
  for i, row in ipairs(rows) do
    ys[i] = y(row) end
  cut = lvl < maxd and best_cut(xcols, rows, ys, mk, m, leaf)
  node = {ys = yadds(ys, mk()), n = #rows}
  if not cut then return node end
  col = cut[1]
  node.ati, node.isnum, node.v = col.at, col.isnum, cut[2]
  yes, no = {}, {}
  for _, row in ipairs(rows) do
    branch = goes_left(row.labels[node.ati], node.v, node.isnum) and yes or no
    branch[#branch + 1] = row
  end
  lvl = lvl + 1
  node.yes = tree(yes, y, xcols, mk, m, leaf, maxd, lvl)
  node.no  = tree(no,  y, xcols, mk, m, leaf, maxd, lvl)
  return node end

function leaf_of(node, row)
  local w,ok
  while node.ati do
    w = row.labels[node.ati]
    ok = goes_left(w, node.v, node.isnum)
    node = ok and node.yes or node.no end
  return node end

-- Forest -------------------------------------------------------------------
function capped(rows, y, n, nk)
  local h,out,u,v
  if not nk then
    u = shuffle(rows)
    out = {}
    for i = 1, min(n, #u) do
      out[#out + 1] = u[i] end
    return out end
  h, out = {}, {}
  for _, row in ipairs(shuffle(rows)) do
    v = y(row)
    h[v] = (h[v] or 0) + 1
    if h[v] <= n then
      out[#out + 1] = row end end
  return out end

function forest(tb, rows, y, mk)
  local xcols,nk,m,half,out,sample = tb.cols.x
  mk = mk or function() return Ys(tb.nk) end
  nk = mk().nk
  if the.rf then
    m = math.ceil(nk and sqrt(#xcols) or #xcols / 3) end
  if the.cap then
    rows = capped(rows, y, the.cap, nk)
  end
  half = function()
    local shuffled,out = shuffle(rows), {}
    for i = 1, floor(#rows / 2) do
      out[i] = shuffled[i] end
    return out
  end
  out = {}
  for i = 1, the.trees do
    sample = the.trees == 1 and rows or half()
    out[i] = tree(sample, y, xcols, mk, m, the.leaf, the.maxd) end
  return out end

function vote(trees, row)
  local nk,h,w,ys = trees[1].ys.nk, nil, #trees
  h = zeros(nk)
  for _, node in ipairs(trees) do
    ys = leaf_of(node, row).ys
    for i = 1, nk do
      h[i] = h[i] + (ys.has[i] / ys.n / w) end end
  return h end

function votes(trees, rows, y)
  local out = {}
  for _, row in ipairs(rows) do
    out[#out + 1] = {y(row), vote(trees, row)} end
  return out end

function predict(trees, row)
  local s = 0
  for _, node in ipairs(trees) do
    s = s + mid(leaf_of(node, row).ys) end
  return s / #trees end

-- AUC ----------------------------------------------------------------------
function auc(pos, neg)
  local s = 0
  if #pos == 0 or #neg == 0 then return nil end
  table.sort(neg)
  for _, p in ipairs(pos) do
    s = s + ((bisect(neg, p) + bisect(neg, p, true)) / 2) end
  return s / (#pos * #neg) end

function aucs(wps)
  local n,sum,pos,neg,scores,a = 0, 0
  for c = 1, #wps[1][2] do
    pos, neg = {}, {}
    for _, wp in ipairs(wps) do
      scores = wp[1] == c and pos or neg
      scores[#scores + 1] = wp[2][c] end
    a = auc(pos, neg)
    if a then
      n = n + #pos
      sum = sum + #pos * a end end
  return 100 * sum / max(n, 1) end

return _ENV
