# hi: source as tiny-type, N-column, colored html (a2ps for the
# browser). Rules picked by suffix: .lisp or .yue. One pass per file:
# named-group alternation, first rule wins, so strings beat comments
# beat keywords. A banner line (";;;; name ---" in lisp, "---- name"
# in yue) starts a block with the next pale background.
# Usage: python3 hi.py COLS FILE... > index.html
import re, sys, html

LISP_KW = ("defun defmacro defmethod defm defstruct defvar defparameter "
      "defpackage let if when unless cond case and or not loop dolist "
      "dotimes lambda setf incf push progn aif flet labels in-package "
      "load return-from destructuring-bind")
YUE_KW = ("and break continue do else elseif export false for from "
      "global if import in local nil not or return switch then true "
      "unless until when while with")
PY_KW = ("def class return if elif else for while in not and or is "
      "None True False import from global try except with as lambda "
      "yield pass continue break")
RULES = dict(
  py  =[("str", r'"""(?:[^\\]|\\.)*?"""|"(?:\\.|[^"\\])*"'),
        ("com", r"#.*"),
        ("fun", r"(?<=^def )\w+|(?<=^class )\w+|(?<=^  def )\w+"),
        ("kw",  r"(?<![\w.])(?:%s)(?![\w])" %
                "|".join(PY_KW.split())),
        ("num", r"(?<![\w.])\d+\.?\d*(?![\w])")],
  lisp=[("str", r'"(?:\\.|[^"\\])*"'),
        ("com", r";.*"),
        ("fun", r"(?<=\(defun )[^\s()]+|(?<=\(defm )[^\s()]+"
                r"|(?<=\(defmacro )[^\s()]+|(?<=\(defstruct )[^\s()]+"),
        ("key", r"(?<![\w-]):[\w=-]+"),
        ("kw",  r"(?<![\w-])(?:%s)(?![\w-])" %
                "|".join(map(re.escape, LISP_KW.split()))),
        ("num", r"(?<![\w-])\d+\.?\d*(?![\w-])")],
  yue =[("str", r'"(?:\\.|[^"\\])*"'),
        ("com", r"--.*"),
        ("fun", r"^[A-Za-z_]\w*(?= *= *(?:\([^)]*\) *)?->)"),
        ("key", r"(?<![\w.]):\w+|\w+(?=:)"),         # :pun and key:
        ("kw",  r"(?<![\w.])(?:%s)(?![\w])" %
                "|".join(map(re.escape, YUE_KW.split()))),
        ("num", r"(?<![\w.])\d+\.?\d*(?![\w])")])
COLOR = dict(str="#a31515", com="#6a737d", fun="#795e26",
             kw="#0033b3",  num="#098658", n="#b0b0b0", key="#8250df")
BG  = ["#fff4e0","#eaf8ec","#f1ecfd","#e2f6f6",
       "#fdfbd8","#fdecf3","#f1f1f1","#ffe9e2"]

def banner(line, lang):
  if lang == "lisp": return line.startswith(";;;;")
  if lang == "yue":  return line.startswith("----")
  return line.startswith("# ") and line[2:3].isupper()

def paint(path):
  lang = ("yue" if path.endswith(".yue") else
          "py" if path.endswith(".py") else "lisp")
  pat  = re.compile("|".join("(?P<%s>%s)" % kv for kv in RULES[lang]), re.M)
  src  = open(path).read()
  out, i = "", 0     # match raw text, escape after: "<=" must not hit "--.*"
  for m in pat.finditer(src):
    out += html.escape(src[i:m.start()], quote=False)
    out += '<span style="color:%s">%s</span>' % (
             COLOR[m.lastgroup or "n"], html.escape(m.group(), quote=False))
    i = m.end()
  out += html.escape(src[i:], quote=False)
  print('<h2 style="font:italic 11pt Times New Roman,serif;margin:.4em 0">'
        '%s</h2>' % path.split("/")[-1])
  print('<pre style="font:6.5pt/1.4 JetBrains Mono,Monaco,Menlo,monospace;'
        'color:#1f2328;columns:%s 80ch;column-gap:2em;'
        'column-rule:1px solid #dddddd">' % cols)
  bg, k = "#ffffff", 0
  for n, (raw, line) in enumerate(zip(src.split("\n"), out.split("\n")), 1):
    if banner(raw, lang): bg, k = BG[k % len(BG)], k + 1
    print('<span style="display:block;background:%s">'
          '<span style="color:%s">%4d</span> %s</span>'
          % (bg, COLOR["n"], n, line), end="")
  print("</pre>")

cols, paths = sys.argv[1], sys.argv[2:]
print('<meta charset=utf-8><title>enuf</title>'
      '<meta name=viewport content="width=device-width,initial-scale=1">'
      '<link rel=stylesheet href="https://fonts.googleapis.com/css2'
      '?family=JetBrains+Mono:wght@400&display=swap">'
      '<style>@page{size:letter landscape;margin:.4in}'
      'html{-webkit-text-size-adjust:100%;text-size-adjust:100%}'
      '@media screen and (max-width:700px){body{margin:4px}'
      'pre{columns:1!important;font-size:calc(100vw/50)!important;'
      'white-space:pre-wrap;overflow-wrap:anywhere}}</style>'
      '<body style="background:#ffffff">')
for p in paths:
  paint(p)
