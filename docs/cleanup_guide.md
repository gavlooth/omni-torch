# Cleanup Module Guide

`lib/data/clean.omni` contains practical preprocessing operations for row-major
numeric tables.

## 1) Replace missing values

```lisp
(define x (tensor/reshape (tensor/arange 0 20 1) (list 4 5)))
(clean/replace-missing-constant x 0.0)
(clean/replace-missing-mean x)
(clean/replace-missing-median x)
(clean/replace-missing-most-frequent x)
```

## 2) Detect and inspect duplicates

```lisp
(clean/duplicate-row-indices x)
(clean/remove-duplicate-rows x)
```

Duplicate detection checks row equality in row-major form.

## 3) Outlier detection

```lisp
(clean/detect-outliers-z-score x 3.0)
(clean/detect-outliers-iqr x)
```

These return masks that can be used as relation-bridge filters downstream.

## 4) Clipping helpers

```lisp
(clean/clip x 0.0 10.0)
(clean/clip-quantile x 0.05 0.95)
```

`clip-quantile` computes quantile bounds and clamps accordingly.

## 5) Label remapping and class diagnostics

```lisp
(define labels (tensor/reshape (tensor/arange 0 12 1) (list 6 2)))
(clean/remap-class-labels labels)
(clean/class-distribution-report labels)
```

## 6) Profile and validation

```lisp
(clean/feature-profile-before-cleanup x)
(clean/feature-profile-after-cleanup x)
(clean/validate-clean-target x)
```

Profiles report finite counts, missing counts/ratios, and per-feature summary
statistics for quick pre/post checks.
