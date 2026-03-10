# Statistics and Cleanup API Freeze Contract

**Status:** Active public API snapshot
**Version:** 2026-03-10

This document freezes the canonical public function names for statistics and cleanup
modules for this milestone.

## Stable statistics exports (canonical)

- `describe/count`
- `describe/minimum`
- `describe/minimum-dim`
- `describe/maximum`
- `describe/maximum-dim`
- `describe/mean`
- `describe/mean-dim`
- `describe/variance`
- `describe/variance-dim`
- `describe/std`
- `describe/std-dim`
- `robust/median`
- `robust/median-dim`
- `robust/quantile`
- `robust/quantile-dim`
- `robust/interquartile-range`
- `robust/interquartile-range-dim`
- `robust/median-absolute-deviation`
- `robust/median-absolute-deviation-dim`
- `distribution/histogram`
- `distribution/skewness`
- `distribution/skewness-dim`
- `distribution/kurtosis`
- `distribution/kurtosis-dim`
- `association/covariance`
- `association/correlation`
- `association/covariance-dim`
- `association/correlation-dim`
- `missing/count`
- `missing/count-dim`
- `missing/sum`
- `missing/sum-dim`
- `missing/minimum`
- `missing/minimum-dim`
- `missing/maximum`
- `missing/maximum-dim`
- `missing/mean`
- `missing/mean-dim`
- `missing/variance`
- `missing/variance-dim`
- `missing/std`
- `missing/std-dim`
- `stats/describe/*`
- `stats/robust/*`
- `stats/distribution/*`
- `stats/association/*`
- `stats/missing/*`

No additional top-level statistical names should be introduced without a new freeze
revision.

## Stable cleanup exports (canonical)

- `clean/replace-missing-constant`
- `clean/replace-missing-mean`
- `clean/replace-missing-median`
- `clean/replace-missing-most-frequent`
- `clean/duplicate-row-indices`
- `clean/remove-duplicate-rows`
- `clean/remap-class-labels`
- `clean/class-distribution-report`
- `clean/detect-outliers-z-score`
- `clean/detect-outliers-iqr`
- `clean/clip`
- `clean/clip-quantile`
- `clean/feature-profile-before-cleanup`
- `clean/feature-profile-after-cleanup`
- `clean/validate-clean-target`
- `clean/_validate-row-major` (internal)

### Backward compatibility policy

- Existing names above remain stable for this milestone.
- Internal symbols not listed are not part of the frozen API and may change.
- Any renaming requires a major revision note in this file plus TODO/update
  tracking.
