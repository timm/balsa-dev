# reg60: the ensemble barely helps, and LightGBM owns big data

Scored on `rel_X = mae_X / mae_mean`, so 1.00 is no better than always
predicting the mean, and **smaller is better**. 59 data sets, 10 folds
(5x2), three treatments: `balsa1` (one tree), `balsa30` (30 trees, rf), and
`lgbm` (LightGBM, 100 trees). The mean-predictor is the divisor, not a
treatment.

Raw MAE is useless for this: regression targets have different units, so a
decomposition on raw MAE gives the data set 90% and the learner 0.0%, which
only says that `cal.housing` has bigger numbers than `bolts`. Everything
below uses `rel_`.

## 1. By group

The split of the classification figure, as far as this corpus allows.
*numeric* = every x column numeric, under 1000 rows. *mixed* = at least one
symbolic column, under 1000 rows. *large, mixed* = 1000 rows or more, any
type. There is no *symbolic* row: only `sensory` and `servo` have no
numeric column at all, and both sit in *mixed*.

| group | sets | b1 | b30 | lgbm | best: b1 | b30 | lgbm |
|---|---|---|---|---|---|---|---|
| numeric | 17 | 0.71 | **0.65** | 0.80 | 5 | **9** | 3 |
| mixed | 24 | 0.78 | **0.68** | 0.78 | 5 | **19** | 9 |
| large, mixed | 18 | 0.67 | 0.61 | **0.37** | 0 | 1 | **18** |
| total | 59 | 0.73 | **0.65** | 0.66 | 10 | 29 | **30** |

The total reads as a dead heat (29 against 30) and is averaging two
opposite regimes. LightGBM wins **18 of 18** large sets and is beaten on
both small groups, where it is level with a single tree (0.78 v 0.78 on
mixed) and worse than one (0.80 v 0.71 on numeric). It fails to beat the
mean-predictor at all on **20 of 59** sets; the median size of those is 61
rows, against 576 for the rest.

## 2. The variance decomposition

Two-way, data set x learner, 10 folds per cell, eta-squared, over all 59:

    data 75    learner 1    fold 6    other 17

Data owns about **58x** more of the variance than the learner. But *other*
(interaction) is **17%**, against 4% for the classification study, which is
the term that says *which* learner wins depends on the data set. Here it
does. The classification claim -- "they are close together almost
everywhere" -- is not available on this corpus.

## 3. The open question: why does the ensemble barely help?

    1 tree -> 30 trees:  0.727 -> 0.648   = 11% less error
    30 trees beat 1 tree on 45 of 59 sets

and that 11% is flat across every size band (+11, +13, +8, +10). Compare
classification, where the same step is worth **+9.8 AUC**, closing 44% of
the gap to a perfect score. Bagging is supposed to be a variance killer.
Here it is worth a tenth.

### A specific hypothesis, cheap to test

We optimise for one loss and score with another. In `balsa.yue`:

    mid = (ys) ->              -- line ~133
      if not ys.nk
        return ys.mu           -- <-- a leaf predicts its MEAN

    predict = (trees, row) ->  -- line ~317
      s = 0
      for node in *trees
        s += mid leaf_of(node, row).ys
      s / #trees               -- <-- and we average across trees

Both steps take a **mean**. The mean is the minimiser of *squared* error.
The minimiser of *absolute* error is the **median**. So the tree is grown
and the ensemble combined for MSE, and then reported as MAE.

That also explains the flatness. Averaging K predictors cuts variance by
about 1/K **in squared error**; it carries no such guarantee for absolute
error, so the ensemble's main mechanism is being measured with a ruler that
cannot see it.

### Three things to run

1. **Swap both means for medians.** `mid` returns the leaf's median y;
   `predict` returns the median across trees. If the hypothesis holds, the
   1 -> 30 gain should grow well past 11%. This is the one to do first: two
   lines, and it either moves or it does not.
2. **Score the current code on RMSE as well as MAE.** If the ensemble gain
   is large on RMSE and small on MAE, that settles it without touching the
   learner at all. Cheaper than 1, and a cleaner test, since nothing in the
   model changes.
3. **Check the leaves are not already pure.** `the.leaf = 2` and
   `maxd = 30` on a 60-row data set gives leaves of one or two rows, where
   mean and median coincide and there is no variance left for bagging to
   average away. If so the story is about tree depth, not about the loss.

If 2 shows a big RMSE gain and a small MAE gain, the headline is not "the
ensemble does not work" but "we have been reporting the wrong loss".
