# Omni Torch Todo

## Purpose

This file is the execution backlog for the current build. It is not the full product roadmap.

Rules for items in this file:
- Each task must contribute directly to the next milestone.
- Each task should be concrete enough to finish without inventing new scope mid-flight.
- Each milestone has explicit exit criteria.
- Optimization, Docker hardening, and release process work stay out of the active backlog until the core behavior is stable.

Status markers:
- `[ ]` not started
- `[-]` in progress
- `[x]` done

Priority markers:
- `P0` required to complete the milestone
- `P1` important but can follow after the milestone core is working
- `P2` later hardening or convenience work

## Milestone Order

1. `M1` Semantic Core
2. `M2` Executor Core
3. `M2b` Libtorch Backend Integration
4. `M3` Statistics Surface
5. `M4` Data Cleanup and Pipelines
6. `M5` Deduce Integration
7. `M6` Hardening, Performance, Docker, and Release

## M1. Semantic Core

Goal:
Lock the execution semantics before expanding the public surface area.

Exit criteria:
- A formal semantics section exists in `docs/FULL_ML_LIBRARY_PROPOSAL.md`.
- The two execution lanes are defined with examples.
- Core behavior is specified for duplicates, null or missing values, type promotion, deterministic ordering, and error handling.
- Each semantic rule has at least one example and one test target listed.

Tasks:
- [x] `P0` Define and document the two execution lanes: `logic semantics` and `analytic semantics`.
- [x] `P0` Define and document the two execution lanes: `logic semantics` and `analytic semantics`.
- [x] `P0` Define and document analytic relation semantics for `set behavior`, `duplicate handling`, `null or missing value handling`, `grouped aggregation`, `ordering`, and `window operations`.
- [x] `P0` Define and document deterministic tie breaking rules for ordered analytic operations.
- [x] `P0` Define and document error behavior for invalid grouping keys and invalid window definitions.
- [x] `P0` Define and document type promotion rules for integer and floating point arithmetic.
- [x] `P0` Define and document missing value propagation rules per operator category.
- [x] `P0` Define and document reproducibility requirements and deterministic mode behavior.
- [x] `P0` Add a formal semantics section to `docs/FULL_ML_LIBRARY_PROPOSAL.md` with examples.
- [x] `P1` Add a compatibility policy for future keyword additions with no silent behavior changes.
- [x] `P0` For each semantic rule, add a linked test target or explicit future test note.

## M2. Executor Core

Goal:
Build the minimal C3 execution substrate needed to run deterministic relational and analytic operations with diagnostics.

Depends on:
- `M1`

Exit criteria:
- The executor skeleton exists in C3.
- The base operator set is available.
- Execution context, cancellation checks, and structured diagnostics exist.
- Planner output can be serialized for debugging and golden tests.

Tasks:
- [x] `P0` Create executor module skeleton in C3 for logical planning and physical execution.
- [x] `P0` Add internal operator interface for scan, filter, project, join, group-by, order-by, and limit.
- [x] `P0` Add execution context object that carries memory limits, thread limits, and diagnostics.
- [x] `P0` Add cancellation and timeout checks at every blocking execution step.
- [x] `P0` Add stable operator identifiers for traceability and debugging.
- [x] `P0` Add planner output serializer for debugging and golden tests.
- [x] `P1` Add failure diagnostics with input relation names and operation names.
- [x] `P1` Add structured execution report output for benchmark and tests.
- [x] `P1` Add operator level timing instrumentation.
- [x] `P2` Add operator level memory instrumentation.

## M2b. Libtorch Backend Integration

Goal:
Make libtorch the explicit execution backend for the first usable Omni Torch surface area instead of leaving backend behavior implicit in scattered shim work.

Depends on:
- `M1`
- `M2`

Exit criteria:
- The backend abstraction boundary between Omni Torch and libtorch is defined.
- Dtype, device, layout, and missing-value mapping rules are documented for the initial supported surface.
- Required operator coverage for the first public stats and cleanup APIs is implemented on libtorch.
- Errors and diagnostics from libtorch are translated into stable Omni-facing behavior.
- Thread and memory controls are wired through the execution context.

Tasks:
- [x] `P0` Define the backend abstraction boundary between the Omni executor and libtorch.
- [x] `P0` Define dtype mapping rules between Omni types and libtorch tensor types.
- [x] `P0` Define device mapping and support policy for CPU and any initial accelerator targets.
- [x] `P0` Define layout and contiguity semantics required at the libtorch boundary.
- [x] `P0` Define missing-value and non-finite value handling rules at the libtorch boundary.
- [x] `P0` Implement the minimal libtorch operator coverage required by the initial statistics surface.
- [x] `P0` Implement the minimal libtorch operator coverage required by the initial cleanup surface.
- [x] `P0` Translate libtorch backend failures into stable Omni diagnostics with operation context.
- [x] `P0` Wire executor thread controls through to libtorch runtime settings.
- [x] `P1` Wire executor memory controls and reporting through the libtorch boundary where feasible.
- [x] `P1` Add backend conformance tests for the initial supported operator set.
- [x] `P1` Document unsupported libtorch behaviors and fallback policy.

## M3. Statistics Surface

Goal:
Expose a small, coherent statistics API on top of the executor and backend shims.

Depends on:
- `M1`
- `M2`
- `M2b`

Exit criteria:
- Required shim operations exist for the first public statistics library.
- `lib/stats` exposes a minimal, consistent API.
- Shape validation and missing-value behavior are defined for public functions.
- The top-level export path is clear.

Tasks:
- [x] `P0` Add shim operation for `is-not-a-number` mask.
- [x] `P0` Add shim operation for `is-infinite` mask.
- [x] `P0` Add shim operation for `is-finite` mask.
- [x] `P0` Add shim operation for `not-a-number to number` replacement with configurable replacement values.
- [x] `P0` Add shim operation for median.
- [x] `P0` Add shim operation for quantile.
- [x] `P0` Add shim operation for covariance.
- [x] `P0` Add shim operation for correlation.
- [x] `P0` Create `lib/stats/describe.omni` with count, minimum, maximum, mean, variance, and standard deviation.
- [x] `P0` Create `lib/stats/robust.omni` with median, quantile, interquartile range, and median absolute deviation.
- [x] `P0` Create `lib/stats/association.omni` with covariance and correlation.
- [x] `P1` Create `lib/stats/distribution.omni` with histogram, skewness, and kurtosis.
- [x] `P1` Create `lib/stats/missing.omni` with missing value aware reductions.
- [x] `P0` Add dimension aware variants for each statistics function in the initial public set.
- [x] `P0` Add keep dimension control for reduction outputs.
- [x] `P0` Add dtype override for numerically stable accumulation.
- [x] `P0` Add exhaustive shape validation and clear error text.
- [x] `P0` Add consistent function naming and re-export from top level `lib/torch.omni`.
- [x] `P2` Add shim operation for histogram.
- [x] `P2` Add shim operation for unique values with counts.
- [x] `P2` Add shim operation for count distinct values.
- [x] `P2` Add shim operation for robust statistics helpers where backend support exists.

## M4. Data Cleanup and Pipelines

Goal:
Support practical preprocessing workflows with explicit fit/transform semantics and leakage prevention.

Depends on:
- `M1`
- `M2`
- `M2b`
- `M3`

Exit criteria:
- `lib/data/clean.omni` exists with the core cleanup operations.
- `lib/data/pipeline.omni` exists with `fit`, `transform`, and `fit and transform`.
- Data split helpers exist for common training flows.
- Leakage-prevention semantics are documented and enforced.

Tasks:
- [x] `P0` Create `lib/data/clean.omni` module skeleton.
- [x] `P0` Add missing value replacement strategy: constant.
- [x] `P0` Add missing value replacement strategy: mean.
- [x] `P0` Add missing value replacement strategy: median.
- [x] `P0` Add clipping helper by fixed bounds.
- [x] `P0` Add clipping helper by quantile bounds.
- [x] `P1` Add missing value replacement strategy: most frequent value.
- [x] `P1` Add outlier detection by z score.
- [x] `P1` Add outlier detection by interquartile range.
- [x] `P1` Add duplicate row detection for row major tensors.
- [x] `P1` Add duplicate row removal for row major tensors.
- [x] `P1` Add class label remapping helper.
- [x] `P1` Add class distribution report helper.
- [x] `P1` Add feature level profile report before cleanup.
- [x] `P1` Add feature level profile report after cleanup.
- [x] `P0` Create `lib/data/pipeline.omni` with pipeline object type.
- [x] `P0` Add `fit` operation for pipelines.
- [x] `P0` Add `transform` operation for pipelines.
- [x] `P0` Add `fit and transform` combined operation.
- [x] `P0` Add deterministic transform ordering guarantees.
- [x] `P1` Add serialization support for fitted pipeline state.
- [x] `P0` Add split helpers for training, validation, and testing datasets.
- [x] `P1` Add stratified split helper for class balanced datasets.
- [x] `P1` Add cross validation split helper.
- [x] `P0` Add data leakage guard checks that enforce fit on training partition only.

## M5. Deduce Integration

Goal:
Connect relation-based reasoning with tensor-based analytics without making semantics opaque.

Depends on:
- `M1`
- `M2`
- `M2b`
- `M4`

Exit criteria:
- The relation contract between deduce and tensor batches is defined.
- Bidirectional bridge helpers exist for selection and summary metadata.
- Explainable rule outputs and lineage metadata exist.

Tasks:
- [x] `P0` Define relation contract between deduce relations and tensor batches.
- [x] `P0` Add relation to tensor selection bridge function.
- [x] `P0` Add tensor to relation summary bridge function for metadata and quality facts.
- [x] `P1` Add rule based data quality checks as reusable relation templates.
- [x] `P1` Add rule based feature gating semantics.
- [x] `P0` Add explainable rule output for accepted and rejected records.
- [x] `P0` Add lineage metadata for derived analytic outputs.
- [x] `P0` Add deterministic ordering rules for logic enriched analytic results.

## M6. Hardening, Performance, Docker, and Release

Goal:
Reduce operational risk, control resource usage, and set release quality bars after the core library behavior is stable.

Depends on:
- `M1`
- `M2`
- `M3`
- `M4`
- `M5`

Exit criteria:
- Core correctness and memory tests are in place.
- Docker-based bounded execution exists for heavy workflows.
- Performance and memory hardening items for known hotspots are addressed.
- Release gates are explicit and enforceable.

Tasks:
- [-] `P0` Add unit tests for every newly added statistics function.
- [-] `P0` Add unit tests for every cleanup strategy.
- [ ] `P0` Add golden tests for deterministic statistics outputs with fixed seeds.
- [-] `P0` Add shape mismatch negative tests for all public operations.
- [-] `P0` Add missing value behavior tests with mixed finite and non finite inputs.
- [ ] `P0` Add integration test for full pipeline fit, transform, train, checkpoint, and restore.
- [x] `P0` Separate host-safe test target from hardening stress test target in the Makefile.
- [x] `P0` Make hardening test container-only by default for host stability.
- [ ] `P1` Add deduce integration tests for logic enriched filtering and traceability.
- [ ] `P1` Add performance regression test for fused operations versus non fused chains.
- [ ] `P1` Add memory regression tests for long pipeline execution.
- [x] `P1` Add container wrapper script for hardening test runs with default 30% CPU and memory limits.
- [ ] `P1` Add fused cleanup path that combines missing value handling and clipping in one backend call.
- [ ] `P1` Add fused statistics path that computes several summary metrics in one backend call.
- [ ] `P1` Add intermediate tensor reuse policy to reduce allocation churn.
- [x] `P1` Add contiguous layout enforcement utility where required.
- [x] `P1` Add fallback strategy for non contiguous views that avoids repeated hidden copies.
- [ ] `P1` Add numeric stability mode that promotes accumulation precision.
- [ ] `P2` Add memory spike detection during chained transformations.
- [ ] `P2` Add guardrails for large quantile workloads with clear warning thresholds.
- [ ] `P2` Add configurable batch processing mode for very large datasets.
- [x] `P2` Add shim operation for in-place clamp.
- [x] `P2` Add shim operation for in-place missing value fill.
- [x] `P2` Add shim operation for thread count control in backend runtime.
- [ ] `P1` Add Docker development image for omni-torch workflows.
- [ ] `P1` Add Docker compose profile with host cpu cap at thirty percent.
- [ ] `P1` Add Docker compose memory cap profile with conservative defaults.
- [ ] `P1` Route benchmark commands through Docker by default.
- [ ] `P1` Route training examples through Docker by default.
- [ ] `P2` Add host side warning if heavy commands are run outside Docker.
- [ ] `P2` Add documented override switch for local direct execution.
- [ ] `P1` Add container startup script that sets backend thread counts based on container limits.
- [x] `P1` Add user guide page for statistics module with complete examples.
- [x] `P1` Add user guide page for cleanup and preparation module with complete examples.
- [x] `P1` Add user guide page for pipeline semantics and data leakage prevention.
- [x] `P1` Add user guide page for deduce integration patterns.
- [ ] `P2` Add cookbook example for tabular classification with cleanup and robust statistics.
- [ ] `P2` Add cookbook example for time series feature extraction with window operations.
- [ ] `P2` Add cookbook example for logic constrained training dataset preparation.
- [ ] `P2` Add migration guide for users moving from direct tensor scripts to pipeline based workflows.
- [x] `P0` Freeze public function names for statistics and cleanup modules.
- [ ] `P0` Freeze semantic rules for grouped aggregation and ordering.
- [ ] `P0` Require all high severity memory and correctness tests to pass.
- [ ] `P0` Require Docker bound benchmark suite to pass within resource limits.
- [ ] `P0` Require documentation coverage for all new public functions.
- [ ] `P0` Publish versioned release notes with behavioral guarantees.

## Current Focus

Active recommendation:
1. Finish `M1` before adding more public library surface.
2. Start `M2` only after the semantics document is concrete enough to write golden tests against.
3. Add `lib/stats/missing.omni` and finish the remaining public stats definitions.
4. Start `M2` implementation once semantics are in place for reproducibility and missing-value propagation.

Immediate next tasks:
- [x] `P0` Write the formal semantics section in `docs/FULL_ML_LIBRARY_PROPOSAL.md`.
- [x] `P0` Resolve deterministic ordering and tie breaking rules.
- [x] `P0` Define missing value propagation and type promotion rules.
- [x] `P0` Create the executor module skeleton in C3.
- [x] `P0` Add planner serialization and stable operator identifiers.
- [x] `P0` Define the Omni executor to libtorch backend boundary.
- [x] `P0` Define dtype, device, and layout mapping rules for the initial supported surface.
