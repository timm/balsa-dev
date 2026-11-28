# regression notes: the ensemble helps a little, LightGBM owns big
# data, TabICL owns everything it can reach

(Updated 2026-11-26 for the Python canon -- tree.py run under
pypy3, reg56 corpus, TabICL arm included. The original lua-era
version of this note is in git history; its hypothesis survives
unchanged, its numbers do not.)

Scored on `rel_X = mae_X / mae_mean`, so 1.00 is no better than
always predicting the mean, and **smaller is better**. 56 data sets,
10 folds (5x2, replayed on recorded row ids), four treatments:
`tree1` (one tree), `tree30` (30 trees, rf = |x|/3 per cut),
`lgbm` (LightGBM, 100 trees), `tabicl` (in-context, on the sets it
finished). The mean-predictor is the divisor, not a treatment.

Raw MAE is useless for this: regression targets have different
units, so a decomposition on raw MAE gives the data set 90% and the
learner 0.0%, which only says that `cal.housing` has bigger numbers
than `bolts`. Everything below uses `rel_`.

## 1. By group

*numeric* = every x column numeric, under 1000 rows. *mixed* = at
least one symbolic column, under 1000 rows. *large* = 1000 rows or
more. (Only `sensory` and `servo` have no numeric column at all;
both sit in *mixed*.) Best = lowest mean rel-MAE on that set.

| group | sets | b1 | b30 | lgbm | tabicl | best: b1 | b30 | lgbm | tabicl |
|---|---|---|---|---|---|---|---|---|---|
| numeric | 17 | 0.71 | **0.64** | 0.71 | 0.52 | 0 | 2 | 0 | **15** |
| mixed | 22 | 0.81 | **0.68** | 0.82 | 0.63 | 1 | 5 | 0 | **16** |
| large | 17 | 0.63 | 0.56 | **0.33** | 0.30 | 0 | 0 | 6 | **11** |
| total | 56 | 0.72 | 0.63 | 0.64 | **0.52** | 1 | 7 | 6 | **42** |

Among the fit-on-this-machine learners the old story holds: LightGBM
wins the large group outright (0.33 against tree30's 0.56) and
loses both small groups, where it is level with a single tree. It
fails to beat the mean-predictor at all on **13 of 56** sets; the
median size of those is 125 rows, against 418 for the rest
(min_child_samples=20 cannot split a ~30-60-row fold, so it
predicts the train mean, silently). TabICL is ranked best on 42 of
56 -- but see runs/reg56-tabicl.txt for its hours-long bill.

## 2. The variance decomposition

Two-way, data set x learner x fold, eta-squared (regvar.py, over
the 100 regression+SE-regression sets TabICL finished):

    non-SE:  data 76   learner  4   fold 11   other  9    (18x)
    SE:      data 83   learner  7   fold  1   other  9    (11x)
    all:     data 83   learner  3   fold  5   other  8    (25x)

Data owns ~25x more variance than the learner. The interaction term
(*other*, ~8-9%) is the one that says *which* learner wins depends
on the data set -- about twice the classification study's, and the
group table above is that term made visible: large vs small is a
regime change.

## 3. The open question: why does the ensemble help so little?

    1 tree -> 30 trees:  0.725 -> 0.631  = 12% less error
    30 trees beat 1 tree on 42 of 56 sets

Compare classification, where the same step is worth ~+10 AUC,
closing roughly half the gap to perfect. Bagging is supposed to be
a variance killer; here it buys an eighth.

### A specific hypothesis, cheap to test

We optimise for one loss and score with another. In `tree.py`:

    class Num:
      def mid(i): return i.mu          # a leaf predicts its MEAN

    class Forest(list):
      def predict(i, row):
        return sum(node.leaf_of(row).ys.mid() for node in i) / len(i)
        #      ^-- and we AVERAGE across trees

Both steps take a **mean**. The mean is the minimiser of *squared*
error; the minimiser of *absolute* error is the **median**. So the
tree is grown and the ensemble combined for MSE, and then reported
as MAE. That also explains the flatness across size bands: averaging
K predictors cuts variance by ~1/K **in squared error**; it carries
no such guarantee for absolute error, so the ensemble's main
mechanism is being measured with a ruler that cannot see it.

### Three things to run

1. **Swap both means for medians.** `Num.mid` returns the leaf's
   median y; `predict` returns the median across trees. If the
   hypothesis holds, the 1 -> 30 gain should grow well past 12%.
   Two lines, and it either moves or it does not.
2. **Score the current code on RMSE as well as MAE.** If the
   ensemble gain is large on RMSE and small on MAE, that settles it
   without touching the learner. Cheaper than 1, and cleaner, since
   nothing in the model changes.
3. **Check the leaves are not already pure.** `the.leaf = 2` and
   `maxd = 30` on a 60-row set gives leaves of one or two rows,
   where mean and median coincide and there is no variance left for
   bagging to average away. If so the story is about tree depth,
   not about the loss.

If 2 shows a big RMSE gain and a small MAE gain, the headline is not
"the ensemble does not work" but "we have been reporting the wrong
loss".
