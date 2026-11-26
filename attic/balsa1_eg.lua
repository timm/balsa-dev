-- vim: set ts=2 sw=2 sts=2 et cc=75:
-- Balsa experiments: lua balsa1_eg.lua [data...]
-- Also accepts --replay FILE, --reg [data...] and --opt [data...].
-- (c) 2026 Tim Menzies <timm@ieee.org>, MIT license
local _ENV = setmetatable({},{__index = require "balsa1"})
local floor, sqrt, max, min, abs = math.floor, math.sqrt,
                                    math.max, math.min, math.abs
local fmt = string.format
local moot = "~/gits/moot/"

-- Stats ------------------------------------------------------------------
function sorted(t)
  local u = {}
  for j, v in ipairs(t) do u[j] = v end
  table.sort(u)
  return u end

function median(t)
  local u = sorted(t)
  return u[1 + floor(#u / 2)] end

function cliffs(xs,ys)
  local d = 0
  for _, x in ipairs(xs) do
    d = d + (bisect(ys,x) - (#ys - bisect(ys,x,true))) end
  return abs(d) / (#xs * #ys) end

function ks(xs,ys)
  local n,m,d = #xs, #ys, 0
  for _, t in ipairs{xs, ys} do
    for _, x in ipairs(t) do
      d = max(d,abs(bisect(xs,x,true) / n - bisect(ys,x,true) / m))
    end end
  return d / sqrt((n + m) / (n * m)) end

function same(xs,ys,  prudence)
  prudence = (prudence == nil) and the.prudence or prudence
  xs, ys = sorted(xs), sorted(ys)
  if abs(adds(xs).mu - adds(ys).mu) < prudence then return true end
  return cliffs(xs,ys) <= 0.195 and ks(xs,ys) <= 1.36 end

function rank(t)
  local r,pool,u = 1, nil, {}
  table.sort(t,function(a,b) return median(a[2]) < median(b[2]) end)
  for _, item in ipairs(t) do
    if pool and not same(pool,item[2]) then r, pool = r + 1, nil end
    pool = pool or {}
    for _, v in ipairs(item[2]) do pool[#pool + 1] = v end
    u[item[1]] = r end
  return u end

-- Rigs -------------------------------------------------------------------
local arms = {{1, false}, {30, true}}

function xval(rows,fn,  m,n)
  local shuffled,train,test,u
  m, n = m or 5, n or 2
  for _ = 1, m do
    shuffled = shuffle(rows)
    for fold = 0, n - 1 do
      train, test = {}, {}
      for j, row in ipairs(shuffled) do
        u = (j - 1) % n == fold and test or train
        u[#u + 1] = row end
      fn(train,test) end end end

function classification(name)
  local tb,at
  tb = Tbl(csv(moot .. "classify/" .. name .. ".csv"))
  at = tb.cols.klass.at
  return tb, function(row) return row.labels[at] end end

function ens(sets)
  local tb,y
  print("#n30 ")
  for _, name in ipairs(sets) do
    tb, y = classification(name)
    xval(tb.rows,function(train,test)
      local start,a = os.clock()
      a = aucs(forest(tb,train,y):votes(test,y))
      io.write(fmt("%s|%.3f |%.3f |\n",name,a,os.clock() - start))
      io.flush() end) end end

function replay(dat)
  local name,tb,y,rows,f,ids,train,test,seen,row,scores,secs,start
  print("#1:q16 n30 ")
  for line in io.lines(dat) do
    if not line:match("^#") then
      f, ids = line:match("^([^|]*)|"), line:match("|([^|]*)$")
      if f ~= name then
        name = f
        tb, y = classification(f)
        rows = {}
        for j, r in ipairs(tb.rows) do rows[#tb.rows - j] = r end end
      train, test, seen = {}, {}, {}
      for id in ids:gmatch("%d+") do
        row = rows[tonumber(id)]
        test[#test + 1], seen[row] = row, true end
      for _, r in ipairs(tb.rows) do
        if not seen[r] then train[#train + 1] = r end end
      scores, secs = {}, {}
      for j, arm in ipairs(arms) do
        the.trees, the.rf = arm[1], arm[2]
        start = os.clock()
        scores[j] = aucs(forest(tb,train,y):votes(test,y))
        secs[j] = os.clock() - start end
      io.write(fmt("%s|%.3f %.3f |%.3f %.3f |%s\n",
                   f,scores[1],scores[2],secs[1],secs[2],ids))
      io.flush() end end end

function row_ids(rows)
  local ids = {}
  for j, row in ipairs(rows) do ids[row] = #rows - j end
  return ids end

function fold_ids(test,ids)
  local u = map(test,function(row) return tostring(ids[row]) end)
  return table.concat(u," ") end

function mae(want,got)
  local s,n = 0, 0
  for j, v in ipairs(want) do
    s = s + abs(v - got[j])
    n = n + 1 end
  return s / max(n,1) end

function reg(sets)
  local dir,tb,goal,y,ids = os.getenv("MOOTREG") or moot .. "regression"
  print("#1:reg n30:reg mean ")
  for _, name in ipairs(sets) do
    tb = Tbl(csv(dir .. "/" .. name .. ".csv"))
    goal = tb.cols.klass
    if not goal then
      for _, col in ipairs(tb.cols.y) do
        if not goal or col.at < goal.at then goal = col end end end
    y = function(row) return row[goal.at] end
    ids = row_ids(tb.rows)
    xval(tb.rows,function(train,test)
      local want,keep,mu,n,scores,secs,v,start,trees,got =
        {}, {}, 0, 0, {}, {}
      for _, row in ipairs(test) do
        v = y(row)
        if v ~= "?" then want[#want + 1], keep[#keep + 1] = v, row end end
      for _, row in ipairs(train) do
        v = y(row)
        if v ~= "?" then mu, n = mu + v, n + 1 end end
      mu = mu / max(n,1)
      for j, arm in ipairs(arms) do
        the.trees, the.rf = arm[1], arm[2]
        start = os.clock()
        trees, got = forest(tb,train,y,Col), {}
        for k, row in ipairs(keep) do got[k] = trees:predict(row) end
        scores[j] = mae(want,got)
        secs[j] = os.clock() - start end
      got = {}
      for j = 1, #keep do got[j] = mu end
      scores[3] = mae(want,got)
      io.write(fmt("%s|%.4g %.4g %.4g |%.3f %.3f |%s \n",name,
                   scores[1],scores[2],scores[3],secs[1],secs[2],
                   fold_ids(test,ids)))
      io.flush() end) end end

function d2h(row)  return row.d2h end

function best5(rows)
  local lo = math.huge
  for j = 1, min(5,#rows) do lo = min(lo,rows[j].d2h) end
  return lo end

function opt(sets)
  local dir,tb,ds,mu,lo,score,ids =
    os.getenv("MOOTOPT") or moot .. "optimize"
  print("#1:opt n30:opt rand ")
  for _, name in ipairs(sets) do
    tb = Tbl(csv(dir .. "/" .. name .. ".csv"))
    ds, lo = {}, math.huge
    for j, row in ipairs(tb.rows) do
      row.d2h = tb:disty(row)
      ds[j] = row.d2h
      lo = min(lo,row.d2h) end
    mu = adds(ds).mu
    score = function(v) return 100 * (1 - (v - lo) / (mu - lo + 1e-32)) end
    ids = row_ids(tb.rows)
    xval(tb.rows,function(train,test)
      local scores,secs,start,trees,guess,keep = {}, {}
      for j, arm in ipairs(arms) do
        the.trees, the.rf = arm[1], arm[2]
        start = os.clock()
        trees, keep = forest(tb,train,d2h,Col), {}
        guess = map(test,function(r) return {r, trees:predict(r)} end)
        table.sort(guess,function(a,b) return a[2] < b[2] end)
        for k = 1, min(5,#guess) do keep[k] = guess[k][1] end
        scores[j] = score(best5(keep))
        secs[j] = os.clock() - start end
      scores[3] = score(best5(shuffle(test)))
      io.write(fmt("%s|%.2f %.2f %.2f |%.3f %.3f |%s \n",name,
                   scores[1],scores[2],scores[3],secs[1],secs[2],
                   fold_ids(test,ids)))
      io.flush() end) end end

local REG = {"2dplanes", "abalone", "ailerons", "auto93", "autohorse",
  "autompg", "autoprice", "bank32nh", "bank8FM", "baskball",
  "bodyfat", "bolts", "breasttumor", "cal.housing", "cholesterol",
  "cleveland", "cloud", "cpu.act", "cpu", "cpu.small",
  "detroit", "diabetes.numeric", "echomonths",
  "elevators", "elusage", "fishcatch", "fried", "fruitfly",
  "gascons", "house16H", "house8L", "housing", "hungarian",
  "kin8nm", "longley", "lowbwt", "machine.cpu", "mbagrade", "meta",
  "mv", "pbc", "pharynx", "pol", "pollution", "puma32H", "puma8NH",
  "pwlinear", "pyrim", "quake", "schlvote", "sensory", "servo",
  "sleep", "stock", "strike", "triazines", "veteran", "vineyard",
  "wisconsin"}

local ENS = {"balance.scale", "blood-transfusion-service-center",
  "breast.w", "diabetes", "ecoli", "glass", "ionosphere", "sonar",
  "vehicle", "wdbc", "vote", "soybean", "tic-tac-toe", "german",
  "car", "solar.flare2", "primary.tumor", "breast.cancer",
  "spect.test", "audiology", "credit.a", "eucalyptus", "ilpd",
  "heart.c", "hepatitis", "colic", "autos", "cylinder.bands",
  "credit.g", "cmc", "hypothyroid", "sick",
  "Internet-Advertisements", "churn", "COMPAS53", "nomao",
  "bank-marketing", "electricity", "adult"}

srand(the.seed)
if arg[1] == "--replay" then return replay(arg[2]) end
if arg[1] == "--opt" then
  table.remove(arg,1)
  return opt(arg) end
if arg[1] == "--reg" then
  table.remove(arg,1)
  return reg(#arg > 0 and arg or REG) end
return ens(#arg > 0 and arg or ENS)
