# Pipeline Semantics and Data Leakage Prevention Guide

This guide documents the intended lifecycle of `lib/data/pipeline.omni` and
how fit/transform semantics prevent leakage from test partitions.

## 1) Pipeline lifecycle

Create, configure, and fit a pipeline explicitly:

```lisp
(define p0 (pipeline/new))
(define p1 (pipeline/with-step p0 clean/replace-missing-mean))
(define p2 (pipeline/fit p1 data-train))
```

Execution:

```lisp
(define data-cleaned (pipeline/transform p2 data-new))
(define fresh (pipeline/fit-transform p0 data-train))
```

## 2) Serialization and restore

```lisp
(define state (pipeline/serialize p2))
(define restored (pipeline/deserialize state))
```

Serialized state carries fit metadata (`ndim`, `columns`, `rows`, `dtype`) used for
subsequent transform validation.

## 3) Fit/transform compatibility checks

Transform always validates:
- input must be tensor
- rank must match fitted rank
- feature count must match if known
- transformed row count must not exceed fit partition rows
- dtype must remain compatible with fit dtype

## 4) Deterministic ordering

All pipeline steps in this milestone are deterministic for identical input tensors
and fixed step lists. This ensures reproducible row alignment when checkpointing
and restoring pipeline state.

## 5) Leakage guard: fit only on training partition

In this design, leak prevention is enforced by requiring fit metadata to originate
from training-derived tensors. Never fit directly on full combined data.

Recommended pattern:

```lisp
(define split (data/split/train-validation-test data 0.8 0.1 0.1))
(define train (ref split 0))
(define val   (ref split 1))
(define test  (ref split 2))

(define prep (pipeline/fit p0 train))
(define val-clean (pipeline/transform prep val))
(define test-clean (pipeline/transform prep test))
```

If you mix partitions (e.g., fit on `train ++ val` and apply only to `test`), you
forfeit leakage guarantees.

## 6) Recommended checkpoints for robust workflows

- Store `state` alongside preprocessing params and split indices.
- Never call `pipeline/fit` in validation or test loops.
- Validate partition compatibility before production serving and after
  checkpoint restore.
