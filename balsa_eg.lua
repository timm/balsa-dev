-- balsa_eg.yue: the dialog. Experiments, stats and printing;
-- balsa.yue is the model it talks to.
-- usage: yue -c -s *.yue, then
--   luajit balsa_eg.lua [data...]        5x2 xval, 30 trees, 16 bins
--   luajit balsa_eg.lua --replay F.dat   rerun F's recorded folds
-- (c) 2026 Tim Menzies <timm@ieee.org>, MIT license
local floor, sqrt, max, min = math.floor, math.sqrt, math.max, math.min
local b = require("balsa")
local the, srand, shuffle, bisect, csv, Tbl, Col, adds, div, disty, forest, votes, predict, aucs = b.the, b.srand, b.shuffle, b.bisect, b.csv, b.Tbl, b.Col, b.adds, b.div, b.disty, b.forest, b.votes, b.predict, b.aucs
local fmt = string.format

-------------------------------------------------------------- stats
local median
median = function(xs)
  local ys
  do
    local _accum_0 = { }
    local _len_0 = 1
    for _index_0 = 1, #xs do
      local x = xs[_index_0]
      _accum_0[_len_0] = x
      _len_0 = _len_0 + 1
    end
    ys = _accum_0
  end
  table.sort(ys)
  return ys[1 + floor(#ys / 2)]
end

local cliffs
cliffs = function(xs, ys)
  local d = 0
  for _index_0 = 1, #xs do
    local x = xs[_index_0]
    d = d + (bisect(ys, x) - (#ys - bisect(ys, x, true)))
  end
  return math.abs(d) / (#xs * #ys)
end

local ks
ks = function(xs, ys)
  local n, m, d = #xs, #ys, 0
  local _list_0 = {
    xs,
    ys
  }
  for _index_0 = 1, #_list_0 do
    local v = _list_0[_index_0]
    for _index_1 = 1, #v do
      local x = v[_index_1]
      d = max(d, math.abs(bisect(xs, x, true) / n - bisect(ys, x, true) / m))
    end
  end
  return d / sqrt((n + m) / (n * m))
end

-- Three questions, and they have to be different ones.  prudence asks if
-- the gap is big enough to bother reporting, in the units we report.
-- cliffs asks about order, ks about shape.  Cohen's d used to sit here too
-- and was dropped: gap over pooled sd is a second opinion on what cliffs
-- already counts, and the part only it supplied -- a floor in raw units --
-- is what prudence now says outright.
local same
same = function(xs, ys, prudence)
  if prudence == nil then
    prudence = the.prudence
  end
  do
    local _accum_0 = { }
    local _len_0 = 1
    for _index_0 = 1, #xs do
      local x = xs[_index_0]
      _accum_0[_len_0] = x
      _len_0 = _len_0 + 1
    end
    xs = _accum_0
  end
  do
    local _accum_0 = { }
    local _len_0 = 1
    for _index_0 = 1, #ys do
      local y = ys[_index_0]
      _accum_0[_len_0] = y
      _len_0 = _len_0 + 1
    end
    ys = _accum_0
  end
  table.sort(xs)
  table.sort(ys)
  if math.abs(adds(xs).mu - adds(ys).mu) < prudence then
    return true
  end
  return cliffs(xs, ys) <= 0.195 and ks(xs, ys) <= 1.36
end

local rank
rank = function(d)
  table.sort(d, function(x, y)
    return median(x[2]) < median(y[2])
  end)
  local r, pool, out = 1, nil, { }
  for _index_0 = 1, #d do
    local kv = d[_index_0]
    if pool and not same(pool, kv[2]) then
      r, pool = r + 1, nil
    end
    pool = pool or { }
    local _list_0 = kv[2]
    for _index_1 = 1, #_list_0 do
      local x = _list_0[_index_1]
      pool[#pool + 1] = x
    end
    out[kv[1]] = r
  end
  return out
end

--------------------------------------------------------------- rigs
local xval
xval = function(rows, fn, m, n)
  if m == nil then
    m = 5
  end
  if n == nil then
    n = 2
  end
  for _ = 1, m do
    local u = shuffle(rows)
    for kf = 0, n - 1 do
      local train, test = { }, { }
      for i, row in ipairs(u) do
        if (i - 1) % n == kf then
          test[#test + 1] = row
        else
          train[#train + 1] = row
        end
      end
      fn(train, test)
    end
  end
end

local ens
ens = function(sets)
  print("#n30 ")
  for _index_0 = 1, #sets do
    local f = sets[_index_0]
    local tb = Tbl(csv("~/gits/moot/classify/" .. tostring(f) .. ".csv"))
    local at = tb.cols.klass.at
    local kl
    kl = function(row)
      return row.labels[at]
    end
    xval(tb.rows, function(train, test)
      local t0 = os.clock()
      local a = aucs(votes(forest(tb, train, kl), test, kl))
      io.write(fmt("%s|%.3f |%.3f |\n", f, a, os.clock() - t0))
      return io.flush()
    end)
  end
end

local replay
replay = function(dat)
  print("#1:q16 n30 ")
  local tb, v, name = nil, nil, nil
  for line in io.lines(dat) do
    local _continue_0 = false
    repeat
      if line:match("^#") then
        _continue_0 = true
        break
      end
      local f = line:match("^([^|]*)|")
      local ids = line:match("|([^|]*)$")
      if f ~= name then
        name = f
        tb = Tbl(csv("~/gits/moot/classify/" .. tostring(f) .. ".csv"))
        v = { }
        for i, row in ipairs(tb.rows) do
          v[#tb.rows - i] = row
        end
      end
      local at = tb.cols.klass.at
      local kl
      kl = function(row)
        return row.labels[at]
      end
      local test, inh = { }, { }
      for s in ids:gmatch("%d+") do
        local row = v[tonumber(s)]
        test[#test + 1] = row
        inh[row] = true
      end
      local train
      do
        local _accum_0 = { }
        local _len_0 = 1
        local _list_0 = tb.rows
        for _index_0 = 1, #_list_0 do
          local row = _list_0[_index_0]
          if not inh[row] then
            _accum_0[_len_0] = row
            _len_0 = _len_0 + 1
          end
        end
        train = _accum_0
      end
      local aucs_, secs = { }, { }
      local _list_0 = {
        {
          1,
          false
        },
        {
          30,
          true
        }
      }
      for _index_0 = 1, #_list_0 do
        local treat = _list_0[_index_0]
        the.trees, the.rf = treat[1], treat[2]
        local t0 = os.clock()
        aucs_[#aucs_ + 1] = aucs(votes(forest(tb, train, kl), test, kl))
        secs[#secs + 1] = os.clock() - t0
      end
      io.write(fmt("%s|%.3f %.3f |%.3f %.3f |%s\n", f, aucs_[1], aucs_[2], secs[1], secs[2], ids))
      io.flush()
      _continue_0 = true
    until true
    if not _continue_0 then
      break
    end
  end
end

local mae
mae = function(want, got)
  local s, n = 0, 0
  for i, w in ipairs(want) do
    s = s + math.abs(w - got[i])
    n = n + 1
  end
  return s / max(n, 1)
end

local _anon_func_0 = function(keep, predict, trees)
  local _accum_0 = { }
  local _len_0 = 1
  for _index_0 = 1, #keep do
    local row = keep[_index_0]
    _accum_0[_len_0] = predict(trees, row)
    _len_0 = _len_0 + 1
  end
  return _accum_0
end
local _anon_func_1 = function(keep, mu)
  local _accum_0 = { }
  local _len_0 = 1
  for _index_0 = 1, #keep do
    local _ = keep[_index_0]
    _accum_0[_len_0] = mu
    _len_0 = _len_0 + 1
  end
  return _accum_0
end
local _anon_func_2 = function(ids, test, tostring)
  local _accum_0 = { }
  local _len_0 = 1
  for _index_0 = 1, #test do
    local row = test[_index_0]
    _accum_0[_len_0] = tostring(ids[row])
    _len_0 = _len_0 + 1
  end
  return _accum_0
end
local reg
reg = function(sets)
  print("#1:reg n30:reg mean ")
  for _index_0 = 1, #sets do
    local f = sets[_index_0]
    local tb = Tbl(csv(tostring(os.getenv("MOOTREG") or "~/gits/moot/regression") .. "/" .. tostring(f) .. ".csv"))
    local g = tb.cols.klass
    if not g then
      local _list_0 = tb.cols.y
      for _index_1 = 1, #_list_0 do
        local col = _list_0[_index_1]
        if not g or col.at < g.at then
          g = col
        end
      end
    end
    local y
    y = function(row)
      return row[g.at]
    end
    local ids = { }
    for i, row in ipairs(tb.rows) do
      ids[row] = #tb.rows - i
    end
    xval(tb.rows, function(train, test)
      local want, keep = { }, { }
      for _index_1 = 1, #test do
        local row = test[_index_1]
        local w = y(row)
        if w ~= "?" then
          want[#want + 1] = w
          keep[#keep + 1] = row
        end
      end
      local mu = 0
      local m = 0
      for _index_1 = 1, #train do
        local row = train[_index_1]
        local w = y(row)
        if w ~= "?" then
          mu = mu + w
          m = m + 1
        end
      end
      mu = mu / max(m, 1)
      local maes, secs = { }, { }
      local _list_0 = {
        {
          1,
          false
        },
        {
          30,
          true
        }
      }
      for _index_1 = 1, #_list_0 do
        local arm = _list_0[_index_1]
        the.trees, the.rf = arm[1], arm[2]
        local t0 = os.clock()
        local trees = forest(tb, train, y, Col)
        maes[#maes + 1] = mae(want, _anon_func_0(keep, predict, trees))
        secs[#secs + 1] = os.clock() - t0
      end
      maes[#maes + 1] = mae(want, _anon_func_1(keep, mu))
      io.write(fmt("%s|%.4g %.4g %.4g |%.3f %.3f |%s \n", f, maes[1], maes[2], maes[3], secs[1], secs[2], table.concat(_anon_func_2(ids, test, tostring), " ")))
      return io.flush()
    end)
  end
end

local _anon_func_3 = function(tb)
  local _accum_0 = { }
  local _len_0 = 1
  local _list_0 = tb.rows
  for _index_0 = 1, #_list_0 do
    local row = _list_0[_index_0]
    _accum_0[_len_0] = row.d2h
    _len_0 = _len_0 + 1
  end
  return _accum_0
end
local _anon_func_4 = function(guess)
  local _accum_0 = { }
  local _len_0 = 1
  for _index_0 = 1, #guess do
    local g = guess[_index_0]
    _accum_0[_len_0] = g[1]
    _len_0 = _len_0 + 1
  end
  return _accum_0
end
local _anon_func_5 = function(ids, test, tostring)
  local _accum_0 = { }
  local _len_0 = 1
  for _index_0 = 1, #test do
    local row = test[_index_0]
    _accum_0[_len_0] = tostring(ids[row])
    _len_0 = _len_0 + 1
  end
  return _accum_0
end
local opt
opt = function(sets)
  print("#1:opt n30:opt rand ")
  for _index_0 = 1, #sets do
    local f = sets[_index_0]
    local tb = Tbl(csv(tostring(os.getenv("MOOTOPT") or "~/gits/moot/optimize") .. "/" .. tostring(f) .. ".csv"))
    local _list_0 = tb.rows
    for _index_1 = 1, #_list_0 do
      local row = _list_0[_index_1]
      row.d2h = disty(tb, row)
    end
    local y
    y = function(row)
      return row.d2h
    end
    local all = adds(_anon_func_3(tb))
    local lo = math.huge
    local _list_1 = tb.rows
    for _index_1 = 1, #_list_1 do
      local row = _list_1[_index_1]
      lo = min(lo, row.d2h)
    end
    local score
    score = function(got)
      return 100 * (1 - (got - lo) / (all.mu - lo + 1e-32))
    end
    local best5
    best5 = function(rows)
      b = math.huge
      for i = 1, min(5, #rows) do
        b = min(b, rows[i].d2h)
      end
      return b
    end
    local ids = { }
    for i, row in ipairs(tb.rows) do
      ids[row] = #tb.rows - i
    end
    xval(tb.rows, function(train, test)
      local scores, secs = { }, { }
      local _list_2 = {
        {
          1,
          false
        },
        {
          30,
          true
        }
      }
      for _index_1 = 1, #_list_2 do
        local arm = _list_2[_index_1]
        the.trees, the.rf = arm[1], arm[2]
        local t0 = os.clock()
        local trees = forest(tb, train, y, Col)
        local guess
        do
          local _accum_0 = { }
          local _len_0 = 1
          for _index_2 = 1, #test do
            local row = test[_index_2]
            _accum_0[_len_0] = {
              row,
              predict(trees, row)
            }
            _len_0 = _len_0 + 1
          end
          guess = _accum_0
        end
        table.sort(guess, function(a, b)
          return a[2] < b[2]
        end)
        scores[#scores + 1] = score(best5(_anon_func_4(guess)))
        secs[#secs + 1] = os.clock() - t0
      end
      scores[#scores + 1] = score(best5(shuffle(test)))
      io.write(fmt("%s|%.2f %.2f %.2f |%.3f %.3f |%s \n", f, scores[1], scores[2], scores[3], secs[1], secs[2], table.concat(_anon_func_5(ids, test, tostring), " ")))
      return io.flush()
    end)
  end
end

local REG = {
  "2dplanes",
  "abalone",
  "ailerons",
  "auto93",
  "autohorse",
  "autompg",
  "autoprice",
  "bank32nh",
  "bank8FM",
  "baskball",
  "bodyfat",
  "bolts",
  "breasttumor",
  "cal.housing",
  "cholesterol",
  "cleveland",
  "cloud",
  "cpu.act",
  "cpu",
  "cpu.small",
  "detroit",
  "diabetes.numeric",
  "echomonths",
  "elevators",
  "elusage",
  "fishcatch",
  "fried",
  "fruitfly",
  "gascons",
  "house16H",
  "house8L",
  "housing",
  "hungarian",
  "kin8nm",
  "longley",
  "lowbwt",
  "machine.cpu",
  "mbagrade",
  "meta",
  "mv",
  "pbc",
  "pharynx",
  "pol",
  "pollution",
  "puma32H",
  "puma8NH",
  "pwlinear",
  "pyrim",
  "quake",
  "schlvote",
  "sensory",
  "servo",
  "sleep",
  "stock",
  "strike",
  "triazines",
  "veteran",
  "vineyard",
  "wisconsin"
}

local ENS = {
  "balance.scale",
  "blood-transfusion-service-center",
  "breast.w",
  "diabetes",
  "ecoli",
  "glass",
  "ionosphere",
  "sonar",
  "vehicle",
  "wdbc",
  "vote",
  "soybean",
  "tic-tac-toe",
  "german",
  "car",
  "solar.flare2",
  "primary.tumor",
  "breast.cancer",
  "spect.test",
  "audiology",
  "credit.a",
  "eucalyptus",
  "ilpd",
  "heart.c",
  "hepatitis",
  "colic",
  "autos",
  "cylinder.bands",
  "credit.g",
  "cmc",
  "hypothyroid",
  "sick",
  "Internet-Advertisements",
  "churn",
  "COMPAS53",
  "nomao",
  "bank-marketing",
  "electricity",
  "adult"
}

srand(the.seed)
if arg[1] == "--replay" then
  return replay(arg[2])
elseif arg[1] == "--opt" then
  table.remove(arg, 1)
  return opt(arg)
elseif arg[1] == "--reg" then
  table.remove(arg, 1)
  return reg(#arg > 0 and arg or REG)
else
  return ens(#arg > 0 and arg or ENS)
end
