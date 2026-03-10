# Deduce Integration Guide

This guide shows how to move between tensor and relation views, apply rule-driven
logic, preserve lineage metadata, and produce deterministic logic-enriched outputs.

## 1) Create a relation from a tensor

```lisp
(define X (tensor/reshape (tensor/arange 0 12 1) (list 4 3)))
(define columns '(c0 c1 c2))
(define rel (deduce/relation X columns))
```

`deduce/relation` validates 2D tensor shape and records:
- row count, column count, and dtype facts
- contract metadata
- initial lineage entry for `tensor->relation`

Inspect basic relation fields:

```lisp
(deduce/relation-columns rel)   ;=> '(c0 c1 c2)
(deduce/relation-rows rel)      ;=> 4
(deduce/relation-cols rel)      ;=> 3
(deduce/relation-source rel)
(deduce/relation-quality rel)
(deduce/relation-lineage rel)
```

## 2) Bridge back to tensors and select rows

```lisp
;; Full tensor extraction
(deduce/relation->tensor rel '())

;; Select rows 1 and 3 from the relation tensor
(define subset (deduce/relation-select rel '(1 3)))
```

`deduce/relation->tensor` expects explicit row indices and `deduce/relation-select`
is a simple alias with naming clarity.

## 3) Convert arbitrary tensor -> relation bridge

When inputs are produced by raw tensor operations, wrap them before applying deduce
rules:

```lisp
(define rel2 (deduce/tensor->deduce-relation X '(c0 c1 c2))
```

Use `deduce/tensor-summary-bridge` when you need a compact metadata-only snapshot:

```lisp
(deduce/tensor-summary-bridge X)
```

## 4) Rule-based quality checks

- Complete rows:

```lisp
(define complete-mask (deduce/rule/complete-row-mask rel))
```

- Column completeness masks:

```lisp
(define complete-column-mask (deduce/rule/complete-column-mask rel))
```

- Row missing ratio / threshold flags:

```lisp
(define ratios (deduce/rule/row-missing-ratio rel))
(define row-ok (deduce/rule/row-missing-threshold? rel 0.2))
```

- Explain decision outcomes (accepted/rejected indices):

```lisp
(define explain (deduce/rule/explain-row-membership rel row-ok))
```

This returns accepted/rejected lists plus counts.

## 5) Deterministic ordering for logic outputs

The relation contract includes deterministic ordering hints. Use
`deduce/rule/logic-output-by-row-order` to keep deterministic record order after
logic filtering:

```lisp
(define filtered-rel (deduce/rule/logic-output-by-row-order rel row-ok))
```

The returned relation carries lineage metadata describing:
- input shape and output shape
- row-order policy used for stable ordering

Retrieve lineage:

```lisp
(deduce/relation-lineage filtered-rel)
```

## 6) Track custom derived outputs

For any derived relation, append lineage explicitly when you create downstream
relations:

```lisp
(define lineage-step
  (deduce/lineage-record 'custom-step rel.rows rel.cols 10 10 '(reason "user-rule") ))
(define rel-with-lineage
  (deduce/lineage-append rel lineage-step))
```

This keeps history for inspection and debugging in logic + analytic hybrid flows.
