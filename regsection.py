#!/usr/bin/env python3 -B
"""regsection: emit the paper's regression section, numbers and all.

Every figure in the prose below is computed here from runs/*.csv, so the
section cannot drift from the runs that produced it -- the habit the
paper argues for in "Count It from the Source, Never Type It In".  The
output is a LaTeX file for \\input, not a fragment to paste.

usage: python3 regsection.py runs/reg56.csv runs/sereg.csv > regression.tex
"""
import csv, os, sys, glob, statistics as st, collections
import regtab, regvar

MOOT = os.path.expanduser("~/gits/moot")


def tex(s):
    return s.replace("_", r"\_")


def fill(s, **kw):
    import re
    for k, v in kw.items(): s = s.replace(f"@@{k}@@", str(v))
    left = set(re.findall(r"@@(\w+)@@", s))
    if left: sys.exit("regsection: nothing to fill @@%s@@ with"
                      % "@@, @@".join(sorted(left)))
    return s


def n_rows(s):
    sub = "optimize" if "/" in s else "regression"
    return sum(1 for l in open(f"{MOOT}/{sub}/{s}.csv") if l.strip()) - 1


def main():
    D = collections.defaultdict(list)
    A, B = set(), set()
    for path, bag in ((sys.argv[1], A), (sys.argv[2], B)):
        for r in csv.DictReader(open(path)):
            D[r["data"]].append(r); bag.add(r["data"])
    G   = regtab.groups(sorted(A), sorted(B))
    ran = {s for s in D if regtab.vals(D, s, "tabicl")}
    skip = sorted(set(D) - ran, key=n_rows)
    N = {s: n_rows(s) for s in D}

    # ---- the mean / ranked-best table -------------------------------
    body = []
    for name, g in G:
        body.append(regtab.line(D, name.replace(", ", ",\\,"), sorted(g))[0])
    body.append(r"\cmidrule{1-11}")
    body.append(regtab.line(D, "total", sorted(D))[0])

    mu_all, win_all, _ = regtab.block(D, sorted(D), regtab.ARMS)
    mu3, _, _ = regtab.block(D, sorted(D), regtab.ARMS[:3])

    # ---- numbers the prose quotes ------------------------------------
    big = sorted(s for s in D if N[s] >= 1000)
    sml = sorted(s for s in D if N[s] < 1000)
    m_big, w_big, _ = regtab.block(D, big, regtab.ARMS[:3])
    m_sml, w_sml, _ = regtab.block(D, sml, regtab.ARMS[:3])
    cfg = sorted(s for s in D if s.startswith("config/"))
    m_cfg, _, _ = regtab.block(D, cfg, regtab.ARMS[:3])
    gain = 100 * (1 - mu3["b30"] / mu3["b1"])
    beat = sum(1 for s in D
               if st.fmean(regtab.vals(D, s, "enuf30"))
               < st.fmean(regtab.vals(D, s, "enuf1")))
    dud = [s for s in D if st.fmean(regtab.vals(D, s, "lgbm")) >= 1.0]
    flat = len(dud)
    flatmed = st.median([N[s] for s in dud])
    restmed = st.median([N[s] for s in D if s not in dud])
    secs = {}
    for p, k in (("runs/reg56.txt", 0), ("runs/sereg.txt", 0)):
        if not os.path.exists(p): continue
        head = open(p).readline().split()[1:]
        for line in open(p):
            if line.startswith("#secs"):
                for h, v in zip(head, line.split()[3:]):
                    secs[h] = secs.get(h, 0) + float(v)
    for p in ("runs/reg56-tabicl.txt", "runs/sereg-tabicl.txt"):
        if not os.path.exists(p): continue
        for line in open(p):
            if line.startswith("#secs"):
                secs["TabICL"] = secs.get("TabICL", 0) + float(line.split()[-1])

    # binary_config is not in the study, but the section says why, with
    # numbers, so it is run once on its own and kept in its own csv.
    bc = collections.defaultdict(list)
    if os.path.exists("runs/binconf.csv"):
        for r in csv.DictReader(open("runs/binconf.csv")):
            bc[r["data"]].append(r)
    bcm = {a: st.fmean([v for s in bc for v in regtab.vals(bc, s, a)])
           for a in ("enuf30", "lgbm")} if bc else {}

    # the learner's own line count, counted from the source, never typed
    import re
    src_ = open("enuf.py").read()
    cap  = int(re.search(r"\bcap=(\d+)", src_).group(1))
    rcap = int(re.search(r"\brcap=(\d+)", src_).group(1))
    cls_ = open(os.path.expanduser(
      "~/gh/jssclose/clsvar.tex")).read()
    m_ = re.search(r"all sets & \d+ & (\d+) & (\d+)", cls_)
    dCls, mCls = m_.group(1), m_.group(2)
    loc = sum(1 for l in open("enuf.py") if not re.match(r"^\s*(#.*)?$", l))

    KW = dict(
        nA=len(A), nB=len(B), nAll=len(D), nRan=len(ran), nSkip=len(skip),
        skip=", ".join("{\\tt %s}" % tex(s.split("/")[-1]) for s in skip),
        table="\n".join(body),
        mu_b1=f"{mu_all['b1']:.2f}", mu_lgbm=f"{mu_all['lgbm']:.2f}",
        mu_b30=f"{mu_all['b30']:.2f}", mu_tab=f"{mu_all['TabICL']:.2f}",
        w_b1=win_all["b1"], w_lgbm=win_all["lgbm"],
        w_b30=win_all["b30"], w_tab=win_all["TabICL"],
        nBig=len(big), nSml=len(sml),
        big_lgbm=f"{m_big['lgbm']:.2f}", big_b30=f"{m_big['b30']:.2f}",
        sml_lgbm=f"{m_sml['lgbm']:.2f}", sml_b30=f"{m_sml['b30']:.2f}",
        wbig_lgbm=w_big["lgbm"], wbig_b30=w_big["b30"],
        wsml_lgbm=w_sml["lgbm"], wsml_b30=w_sml["b30"],
        all_b1=f"{mu3['b1']:.2f}", all_b30=f"{mu3['b30']:.2f}",
        gain=f"{gain:.0f}", beat=beat, flat=flat,
        flatmed=f"{flatmed:,.0f}", restmed=f"{restmed:,.0f}",
        nCfg=len(cfg), cfg_b1=f"{m_cfg['b1']:.2f}",
        cfg_b30=f"{m_cfg['b30']:.2f}",
        
        sec_b30=f"{secs.get('enuf(30)', 0):,.0f}",
        sec_lgbm=f"{secs.get('lgbm', 0):,.0f}",
        sec_tab=f"{secs.get('TabICL', 0):,.0f}",
        costx=f"{secs.get('TabICL', 0) / max(secs.get('lgbm', 1), 1):,.0f}",
        loc=loc, cap=cap, caphalf=cap // 2, rcap=rcap,
        dCls=dCls, mCls=mCls,
        bcx=f"{bcm['enuf30'] / bcm['lgbm']:.0f}" if bcm else "??",
        bc_lgbm=f"{bcm['lgbm']:.3f}" if bcm else "??",
        bc_b30=f"{bcm['enuf30']:.2f}" if bcm else "??",
    )
    Here = os.path.dirname(os.path.abspath(__file__))
    out = open(os.path.join(Here, "regsection.tmpl")).read()
    # the two figures ship in their own file: they are declared early in
    # main.tex so they can float onto page 2 with Figures 2 and 3
    if len(sys.argv) > 3:
        open(sys.argv[3], "w").write(
            fill(open(os.path.join(Here, "regfigs.tmpl")).read(), **KW))
    # @@name@@, not str.format: LaTeX is made of braces and this file is
    # mostly LaTeX.  An unfilled @@...@@ left in the output is a bug, and
    # the check at the bottom says so rather than shipping it.
    print(fill(out, **KW))


if __name__ == "__main__": main()
