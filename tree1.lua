-- vim: set ts=2 sw=2 sts=2 et cc=75:
-- Enuf learner: CSV data, bins, decision trees, forests, and AUC.
-- (c) 2026 Tim Menzies <timm@ieee.org>, MIT license
-- Keep module functions private while reading standard Lua globals.
local globals = _ENV
local _ENV = setmetatable({},{__index = globals})

local floor, ceil, sqrt = math.floor, math.ceil, math.sqrt
local max,min,log,LOG2 = math.max, math.min, math.log, math.log(2)

the = {seed=1234567891, prudence=0.01, bins=16, trees=30,
       cap=200, leaf=2, maxd=30, rf=true}

-- Random -------------------------------------------------------------------
local seed = the.seed

function srand(n)  seed = n end

function rand(  n)
  seed = (16807 * seed) % 2147483647
  return 1 + floor((n or 1) * seed / 2147483647) end

function shuffle(t)
  local u,k = {t[1]}
  for j = 2, #t do
    k = rand(j)
    u[j] = u[k]
    u[k] = t[j] end
  return u end

function pick(t,  n)
  local u = {}
  n = n and n < #t and n or #t
  if n <= 0 then return u end
  if n < #t then t = shuffle(t) end
  for j = 1, ceil(n) do u[j] = t[j] end
  return u end

function map(t,fn)
  local u={}; for j,v in ipairs(t) do u[j]=fn(v) end; return u end

function sum(t,fn,  iter)
  local s=0;  for _, v in (iter or ipairs)(t) do s=s+fn(v) end
  return s end

-- CSV ----------------------------------------------------------------------
function thing(v)
  v = v:match("^%s*(.-)%s*$");  return tonumber(v) or v end

function csv(file)
  local rows,txt,row = {}
  file = file:gsub("^~",os.getenv("HOME"))
  for line in io.lines(file) do
    txt = line:gsub("\r",""):match("^%s*(.-)%s*$")
    if txt ~= "" then
      row = {}
      for cell in (txt .. ","):gmatch("(.-),") do
        row[#row + 1] = thing(cell) end
      rows[#rows + 1] = row end end
  return rows end

function new(mt,t)  mt.__index = mt;  return setmetatable(t,mt) end

-- Data ---------------------------------------------------------------------
-- Header: Uppercase = Num, else Sym. Last char: ! class, X skip.
-- "?" cells unknown. A row IS its csv cells (numeric indexes),
-- plus one named field: labels, its label per x column.
local Num, Sym, Counts, Data = {}, {}, {}, {}


function Col(  txt,at)
  txt = (txt == nil) and "N" or txt
  at = (at == nil) and 0 or at
  if txt:match("^%u") then
    return new(Num,{at=at, txt=txt, n=0, mu=0, m2=0,
                     goal=txt:match("%-$") and 0 or 1}) end
  return new(Sym,{at=at, txt=txt, n=0, has={}}) end

local function add(i,v,  inc)
  inc = (inc == nil) and 1 or inc
  if v == "?" then return end
  i.n = i.n + inc
  return i:add_known(v,inc) end
Num.add, Sym.add = add, add

function Counts.add(i,v)
  if v > 0 then
    i.has[v] = i.has[v] + 1
    i.n = i.n + 1 end end

function Num.add_known(i,v,inc)
  local d = v - i.mu
  i.mu = i.mu + inc * d / i.n
  i.m2 = max(0,i.m2 + inc * d * (v - i.mu)) end

function Sym.add_known(i,v,inc)  i.has[v] = (i.has[v] or 0) + inc end

function adds(vs,  col)
  col = (col == nil) and Col() or col
  for j = 1, #vs do col:add(vs[j]) end
  return col end

function Num.div(i)  return i.n < 2 and 0 or sqrt(i.m2 / (i.n - 1)) end

function Sym.div(i)
  local term = function(m)
    return m > 0 and -(m / i.n * log(m / i.n)) or 0 end
  return sum(i.has,term,pairs) / LOG2 end

-- class counts: an array indexed by label 1..nk
function zeros(n)
  local u = {}
  for j = 1, n do u[j] = 0 end
  return u end

function Ys(nk)  return new(Counts,{nk=nk, n=0, has=zeros(nk)}) end

function yfold(a,b,  inc,u)
  inc = (inc == nil) and 1 or inc
  u = (u == nil) and a or u
  return a:fold(b,inc,u) end

function Counts.fold(i,b,inc,u)
  for j = 1, i.nk do
    u.has[j] = i.has[j] + inc * b.has[j] end
  u.n = i.n + inc * b.n
  return u end

function Num.fold(i,b,inc,u)
  local n,mu,gap,between = i.n + inc * b.n
  if n <= 0 then
    u.n, u.mu, u.m2 = 0, 0, 0
    return u end
  mu = i.mu + inc * b.n * (b.mu - i.mu) / n
  -- The gap uses the mean of the smaller summary.
  gap = b.mu - (n < i.n and mu or i.mu)
  between = min(n,i.n) * b.n * gap * gap / max(n,i.n)
  u.m2 = max(0,i.m2 + inc * (b.m2 + between))
  u.mu, u.n = mu, n
  return u end

function Counts.div(i)
  local term = function(c) return c > 0 and -(c * log(c)) or 0 end
  return i.n < 2 and 0 or (sum(i.has,term) / i.n + log(i.n)) / LOG2 end

function Num.norm(i,v)
  local z = (v - i.mu) / (i:div() + 1e-32)
  z = max(-3,min(3,z))
  return 1 / (1 + math.exp(-1.7 * z)) end

function Data.disty(i,row)
  local d,n,v = 0, 0
  for _, col in ipairs(i.cols.y) do
    v = row[col.at]
    if v ~= "?" then
      d = d + ((col:norm(v) - col.goal) ^ 2)
      n = n + 1 end end
  return (d / max(n,1)) ^ 0.5 end

function Num.mid(i)  return i.mu end
Sym.mid = Num.mid

function Counts.mid(i)
  local best,most = nil, -1
  for j = 1, i.nk do
    if i.has[j] > most then
      best, most = j, i.has[j] end end
  return best end

function Cols(names)
  local all,x,y,klass,col,suffix = {}, {}, {}, nil
  for at, txt in ipairs(names) do all[at] = Col(txt,at) end
  for j = #all, 1, -1 do
    col = all[j]
    suffix = col.txt:sub(-1)
    if suffix ~= "X" then
      if suffix == "+" or suffix == "-" or suffix == "!" then
        y[#y + 1] = col
        if suffix == "!" then
          klass = col end
      else
        x[#x + 1] = col end end end
  return {names=names, all=all, x=x, y=y, klass=klass} end

-- Labels -------------------------------------------------------------------
-- Known values use positive integers, and "?" is -1. Each column's
-- labels table records the numeric bin boundaries or symbolic values.
function bisect(t,v,  le)
  local lo,hi,mid,below = 1, #t + 1
  while lo < hi do
    mid = floor((lo + hi) / 2)
    below = le and t[mid] <= v or (not le and t[mid] < v)
    if below then lo = mid + 1 else hi = mid end end
  return lo - 1 end

function Num.labels_for(i,n,rows)
  local labels,values,v,last = {}, {}
  for _, row in ipairs(rows) do
    v = row[i.at]
    if v ~= "?" then values[#values + 1] = v end end
  if #values == 0 then return labels end
  table.sort(values)
  for j = 0, n - 1 do
    v = values[1 + floor(j * #values / n)]
    if v ~= last then
      labels[#labels + 1] = v
      last = v end end
  return labels end

function Sym.labels_for(i,n)
  local labels,ranked = {}, {}
  for value, count in pairs(i.has) do
    ranked[#ranked + 1] = {value, count} end
  table.sort(ranked,function(a,b)
    if a[2] == b[2] then
      return tostring(a[1]) < tostring(b[1]) end
    return a[2] > b[2] end)
  for j = 1, min(n - 1,#ranked) do
    labels[j] = ranked[j][1] end
  return labels end

function tolabel(col,v)  return v == "?" and -1 or col:label(v) end

function Num.label(i,v)  return max(1,bisect(i.labels,v,true)) end

function Sym.label(i,v)
  for j, label in ipairs(i.labels) do
    if label == v then return j end end
  return #i.labels + 1 end

function Sym.class_labels(i)  return i:labels_for(math.huge) end

function Data.labelall(i)
  local klass,cols = i.cols.klass, {}
  for j, col in ipairs(i.cols.x) do
    col.labels = col:labels_for(the.bins,i.rows)
    cols[j] = col end
  i.nk = 0
  if klass and klass.class_labels then
    klass.labels = klass:class_labels()
    cols[#cols + 1] = klass
    i.nk = #klass.labels end
  for _, row in ipairs(i.rows) do
    row.labels = {}
    for _, col in ipairs(cols) do
      row.labels[col.at] = tolabel(col,row[col.at]) end end end

function Tbl(rows)
  local tb,row = new(Data,{cols = Cols(rows[1]), rows = {}})
  for j = #rows, 2, -1 do
    row = rows[j]
    tb.rows[#tb.rows + 1] = row
    for _, col in ipairs(tb.cols.all) do
      col:add(row[col.at]) end end
  tb:labelall()
  return tb end

-- Trees --------------------------------------------------------------------
function argmost(better)
  local best,score = nil, nil

  return function(x,s)
    if x and (score == nil or better(s,score)) then
      best, score = x, s end
    return best, score end end

function xpect(a,b)
  return (a.n * a:div() + b.n * b:div()) / (a.n + b.n + 1e-32) end

local function cuts(i,rows,targets,mk,leaf,best)
  local nb,bins,label,prefix,total,rest,left,right = #i.labels, {}
  for b = 1, nb + 1 do bins[b] = mk() end
  for j, row in ipairs(rows) do
    label = row.labels[i.at]
    if label > 0 then
      bins[label]:add(targets[j]) end end
  prefix, total, rest = mk(), mk(), mk()
  for _, bin in ipairs(bins) do yfold(total,bin) end
  for b = 1, nb do
    left, right = i:split_bin(bins[b],prefix,total,rest)
    if leaf <= min(left.n,right.n) then
      best({i, b},xpect(left,right)) end end end
Num.cuts, Sym.cuts = cuts, cuts

function Num.split_bin(i,bin,prefix,total)
  return yfold(prefix,bin), yfold(total,bin,-1) end

function Sym.split_bin(i,bin,_,total,rest)
  return bin, yfold(total,bin,-1,rest) end

function best_cut(xcols,rows,ys,mk,m,leaf)
  local best = argmost(function(a,b) return a < b end)
  for _, col in ipairs(pick(xcols,m)) do
    col:cuts(rows,ys,mk,leaf,best) end
  return best() end

local Node = {}

function Node.goes_left(i,row)
  local label = row.labels[i.ati]
  return label < 0 or i:matches(label) end
Num.node, Sym.node = new(Node,{}), new(Node,{})

function Num.node.matches(i,label)  return label <= i.v end

function Sym.node.matches(i,label)  return label == i.v end

function tree(rows,y,xcols,mk,m,leaf,maxd,  lvl)
  local targets,cut,node,col,yes,no,branch = map(rows,y)
  lvl = (lvl == nil) and 0 or lvl
  cut = lvl < maxd and best_cut(xcols,rows,targets,mk,m,leaf)
  node = new(Node,{ys = adds(targets,mk()), n = #rows})
  if not cut then return node end
  col = cut[1]
  node.ati, node.v = col.at, cut[2]
  new(col.node,node)
  yes, no = {}, {}
  for _, row in ipairs(rows) do
    branch = node:goes_left(row) and yes or no
    branch[#branch + 1] = row end
  lvl = lvl + 1
  node.yes = tree(yes,y,xcols,mk,m,leaf,maxd,lvl)
  node.no  = tree(no,y,xcols,mk,m,leaf,maxd,lvl)
  return node end

function Node.leaf_of(i,row)
  while i.ati do
    i = i:goes_left(row) and i.yes or i.no end
  return i end

-- Forest -------------------------------------------------------------------
local Forest = {}

function capped(rows,y,n,  nk)
  local counts,kept,shuffled,label
  if not nk then
    shuffled = shuffle(rows)
    kept = {}
    for j = 1, min(n,#shuffled) do
      kept[#kept + 1] = shuffled[j] end
    return kept end
  counts, kept = {}, {}
  for _, row in ipairs(shuffle(rows)) do
    label = y(row)
    counts[label] = (counts[label] or 0) + 1
    if counts[label] <= n then
      kept[#kept + 1] = row end end
  return kept end

function forest(tb,rows,y,  mk)
  local xcols,nk,m,half,trees,sample = tb.cols.x
  mk = mk or function() return Ys(tb.nk) end
  nk = mk().nk
  if the.rf then
    m = ceil(nk and sqrt(#xcols) or #xcols / 3) end
  if the.cap then rows = capped(rows,y,the.cap,nk) end

  half = function()
    local shuffled,u = shuffle(rows), {}
    for j = 1, floor(#rows / 2) do u[j] = shuffled[j] end
    return u end
  trees = new(Forest,{})
  for j = 1, the.trees do
    sample = the.trees == 1 and rows or half()
    trees[j] = tree(sample,y,xcols,mk,m,the.leaf,the.maxd) end
  return trees end

function Forest.vote(i,row)
  local nk,h,w,ys = i[1].ys.nk, nil, #i
  h = zeros(nk)
  for _, node in ipairs(i) do
    ys = node:leaf_of(row).ys
    for j = 1, nk do
      h[j] = h[j] + (ys.has[j] / ys.n / w) end end
  return h end

function Forest.votes(i,rows,y)
  return map(rows,function(row) return {y(row), i:vote(row)} end) end

function Forest.predict(i,row)
  local fn = function(node) return node:leaf_of(row).ys:mid() end
  return sum(i,fn) / #i end

-- AUC ----------------------------------------------------------------------
function auc(pos,neg)
  local s = 0
  if #pos == 0 or #neg == 0 then return nil end
  table.sort(neg)
  for _, p in ipairs(pos) do
    s = s + ((bisect(neg,p) + bisect(neg,p,true)) / 2) end
  return s / (#pos * #neg) end

function aucs(wps)
  local n,sum,pos,neg,scores,a = 0, 0
  for c = 1, #wps[1][2] do
    pos, neg = {}, {}
    for _, wp in ipairs(wps) do
      scores = wp[1] == c and pos or neg
      scores[#scores + 1] = wp[2][c] end
    a = auc(pos,neg)
    if a then
      n = n + #pos
      sum = sum + #pos * a end end
  return 100 * sum / max(n,1) end

return _ENV
