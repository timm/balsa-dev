---
name: lean-lua
description: Review or rewrite Lua in the user's compact Enuf style. Use for readability passes, generated Lua cleanup, and comparisons with the hand-edited enuf1.lua.
---

# Lean Lua

Use the Data section of `enuf1.lua` as the style reference. Read `Col`,
`add`, `zeros`, `Ys`, `yadd`, and `yadds` before rewriting.
A compact function should be easy to follow from top to bottom. Each line
should have a visible job, and a short name should keep the same meaning
throughout the operation.

## Conventions

- In this Lua 5.2+ variant, use a private environment with standard-global
  lookup. Define module functions with bare `function name(...)` and finish
  with `return _ENV`. Returning the environment exposes helper functions.
  Keep function scratch state and private module state local.
- Use one local declaration immediately after the function header, as in
  `local got,out,i` in `pick`. Keep assignments where the values become
  meaningful. Let Lua scope its own loop variables. Preserve an inner local
  when shadowing, a fresh binding captured by a closure, or lifetime matters.
  When the declaration mixes several stages, reconsider the function's shape.
- Indent two spaces. Keep the Vim modeline
  `-- vim: set ts=2 sw=2 sts=2 et cc=75:` at the top. Aim for 75 columns.
  Join standalone `end` lines to the preceding line.
- Replace generated accumulator scaffolding with direct loops. Use `out[i]`
  when the loop already supplies the output index, or `out[#out + 1]` to
  append. Share a rule when the reader would otherwise verify it twice.
- In Enuf style passes, edit only `enuf1.lua`. Preserve signatures,
  table fields, row and RNG order, and results. Leave `enuf.lua`,
  `enuf.yue`, `enuf_eg.lua`, and `enuf_eg.yue` untouched.

## Review

Find the lines that force the reader to reconstruct meaning from elsewhere.
In `tree`, follow the split from the pair through `col, b` into the node
fields. Look for repeated recursive arguments that obscure what changes.
In `yfold`, trace the sign of `inc`, the possible alias between `out` and
`ys1`, and the change of reference mean. Its class-count update is a good
example of compact code whose intent is visible on the line.

When asked to measure leanness, count characters on trimmed, nonblank code
lines, skipping full-line comments. Report code lines and lines over 75
columns alongside that count. Counts of explicit local declarations and
standalone ends can show where the reduction came from. Read the changed
functions against the examples above before judging a style pass.
