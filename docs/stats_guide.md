# Statistics Module Guide

The statistics surface is organized through thin wrappers in `lib/stats/*` over
`lib/torch.omni` and missing-aware helpers in `stats/missing.omni`.

## 1) Basic descriptors

```lisp
(define t (tensor/arange 0 100 1))
(describe/count t)
(describe/minimum t)
(describe/maximum t)
(describe/mean t)
(describe/variance t)
(describe/std t)
```

Dimension aware:

```lisp
(describe/mean-dim t 0 true)
(describe/variance-dim t 1 false)
```

## 2) Robust stats

```lisp
(robust/median t)
(robust/quantile t 0.95)
(robust/interquartile-range t)
(robust/median-absolute-deviation t)
```

## 3) Missing-aware helpers

```lisp
(define x (tensor/reshape (tensor/arange 0 12 1) (list 3 4)))
(stats/missing/count x)
(stats/missing/sum x)
(stats/missing/mean x)
(stats/missing/variance x)
(stats/missing/std x)
(stats/missing/mean-dim x 0 true)
```

The `stats/missing/*` API uses explicit mask-and-replace behavior so you can
separate reduction intent from missingness handling.

## 4) Distribution helpers

```lisp
(distribution/histogram t 10 (tensor/item (tensor/min-val t)) (tensor/item (tensor/max-val t)))
(distribution/skewness t)
(distribution/kurtosis t)
(distribution/skewness-dim t 0 false)
```

## 5) Association helpers

```lisp
(define a (tensor/reshape (tensor/arange 0 100 1) (list 50 2)))
(association/covariance a a)
(association/correlation a a)
(association/covariance-dim a a 0 true)
(association/correlation-dim a a 0 false)
```

## 6) Implementation notes

- Robust and descriptive ops are exported both as module-level and `stats/` names in
  `lib/torch.omni`.
- Use dim-aware variants when reducing across known axes for explicit shape control.
