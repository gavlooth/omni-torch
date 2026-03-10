# Omni Torch Full ML Library Proposal

## 1. Objective
Build `omni-torch` into a full ML library for Omni Lisp that supports:
- tensor math and autograd
- model definition (layers/modules)
- training loops and optimizers
- data pipelines and batching
- inference and checkpointing
- optional scaling (mixed precision, multi-device, distributed)

Target: a practical library that can train small to medium models on CPU first, then scale to CUDA.

## 2. Current Baseline (already in repo)
Existing foundation is strong:
- C++ shim over libtorch in `csrc/torch_shim.cpp` + FFI declarations in `lib/ffi/torch.omni`
- high-level tensor wrapper in `lib/torch.omni`
- broad operator surface (shape ops, reductions, activation functions, indexing, FFT, conv1d, embedding, dropout)
- examples (`examples/xor_nn.omni`, diffusion demos)

Current main gap: this is mostly an operator binding package, not yet a full end-to-end ML framework.

## 3. Product Scope

### In scope
- ergonomic tensor + autograd APIs
- `nn` module system (parameter registration, train/eval mode)
- optimizer suite
- losses and metrics
- dataloading and dataset utilities
- serialization/checkpoint API
- reproducibility utilities (seed, determinism modes)
- performance profiling and benchmark harness

### Out of scope for v1
- custom CUDA kernel authoring DSL
- TPU/XLA backend
- graph compiler equivalent to TorchDynamo/TorchInductor

## 4. Target Architecture

### 4.1 Layered design
1. Backend layer (C++/FFI)
- Owns libtorch interop and low-level memory/resource safety.
- Stable C ABI surface for Omni.

2. Core tensor/runtime layer
- Tensor handle abstraction.
- dtype/device/shape validation.
- deterministic resource cleanup policy.

3. Autograd + module layer
- gradient-enabled tensors.
- `Module` protocol for parameters/submodules/mode switching.

4. Training layer
- optimizers, losses, schedulers, loop helpers.

5. Data + IO layer
- datasets, transforms, collate functions, shuffling/sampling.

6. Model zoo + application layer
- canonical models and training recipes.

### 4.2 Proposed package layout

```
lib/
  torch.omni                  ; top-level API re-export
  core/
    tensor.omni               ; tensor type + metadata + generic ops
    dtype.omni                ; dtype/device constants + checks
    shape.omni                ; shape utility helpers
    random.omni               ; seed + RNG API
    memory.omni               ; handle lifetime + leak guards
  autograd/
    grad.omni                 ; requires-grad, backward, no-grad
    functional.omni           ; differentiable functional ops
  nn/
    module.omni               ; Module protocol and registry
    parameter.omni            ; Parameter type and helpers
    layers/
      linear.omni
      conv1d.omni
      norm.omni
      embedding.omni
      dropout.omni
      activation.omni
    init.omni                 ; xavier/kaiming/normal/uniform
  optim/
    sgd.omni
    adam.omni
    adamw.omni
    rmsprop.omni
    scheduler.omni            ; step/cosine/warmup
  losses/
    regression.omni           ; mse/mae/huber
    classification.omni       ; cross-entropy/nll/bce
  data/
    dataset.omni              ; dataset protocol
    dataloader.omni           ; batching/prefetch/shuffle
    sampler.omni              ; sequential/random/weighted
    transforms.omni           ; compose/map/normalize/tokenize hooks
  metrics/
    classification.omni
    regression.omni
  io/
    checkpoint.omni           ; save/load state_dict and optimizer state
    format.omni               ; tensor-safe serialization codecs
  train/
    loop.omni                 ; fit/evaluate/predict
    callbacks.omni            ; early-stop/checkpoint/logging
    history.omni              ; structured metrics history
  dist/
    device.omni               ; cpu/cuda selection
    parallel.omni             ; data-parallel utilities (phase 2+)
```

## 5. API Direction (Omni-first)

### 5.1 Core tensor ergonomics
- keep current `Tensor` type and dispatched arithmetic
- add explicit device API:
  - `(tensor/to-device t 'cpu)`
  - `(tensor/to-device t 'cuda)`

### 5.2 Module API
Proposed user surface:

```lisp
(define model
  (nn/sequential
    (nn/linear 784 256)
    (nn/relu)
    (nn/dropout 0.1)
    (nn/linear 256 10)))

(nn/train! model)
(define y (model x))
```

### 5.3 Training API

```lisp
(define opt (optim/adamw (nn/parameters model) 0.001))
(define loss (loss/cross-entropy logits targets))
(autograd/backward loss)
(optim/step! opt)
(optim/zero-grad! opt)
```

### 5.4 High-level loop API

```lisp
(train/fit model train-loader
  {'epochs 20
   'optimizer (optim/adamw (nn/parameters model) 0.0003)
   'loss-fn loss/cross-entropy
   'callbacks (list (train/early-stop {'patience 3})
                    (train/checkpoint "chkpt.omni"))})
```

## 6. Reliability and Performance Requirements

### 6.1 Resource and runtime constraints
All heavy workflows should remain Docker-bound with capped host usage.
- Local dev image: `docker/omni-torch-dev`
- Suggested default caps:
  - CPU: max 30% host (`--cpus="0.3"` or equivalent compose limit)
  - Memory: fixed ceiling (for example `--memory="4g"`, tune by host class)
- No heavy benchmark/train scripts running directly on host by default.

### 6.2 Correctness guardrails
- strict shape and dtype checks at API boundaries
- deterministic seed API and optional deterministic backend mode
- leak checks in long-running train loops
- operation parity tests against libtorch ground truth on small tensors

### 6.3 Benchmarking
- microbench: operator latency and throughput
- macrobench: reference training jobs (XOR, tiny CNN, tiny transformer)
- report wall time, peak memory, and step/sec

## 6.4 Semantics Contract (M1)

This section is the normative baseline for all operators and executors.

### 6.4.1 Execution lanes
- `logic semantics`: row/record oriented transformations, filtering, projection, relation composition, and rule-based validation.
- `analytic semantics`: tensor-oriented statistical/reduction operators, ordering, window operations, and numeric transforms.
- A pipeline must never mix lane metadata implicitly. Conversions between lanes are explicit through bridge functions and must preserve provenance.

### 6.4.2 Set behavior
- Duplicate records in input relations are preserved unless a relation explicitly requests deduplication.
- Grouped operations operate on all rows in input order by default.
- Null or missing values are considered ordinary rows unless an operator declares filtering of missing/masked values.

### 6.4.3 Missing and invalid value behavior
- Per-operator behavior must be declared for each argument axis: input, accumulator, and output.
- For numeric ops, standard floating behavior is expected (`NaN` stays in `NaN` lineage unless overwritten by an explicit fill strategy).
- `is-finite`, `is-inf`, and `is-nan` masks are first-class operators and must return mask tensors with original shape.
- Missing value conversions must be explicit with a documented replacement value and policy.

### 6.4.3a Missing-value propagation by operator category
- Scan operators:
  - propagate missing row markers as regular rows; missing metadata is preserved on pass-through.
  - no implicit fill; missingness status is stable unless a fill/rewrite operator is invoked.
  - test target note: `test-semantic-scan-missing-pass-through`.
- Filter operators:
  - rows are dropped only by explicit predicate result `false`/`0`.
- predicate yields missing/NaN, missing is treated as `false` for boolean masking unless the operator explicitly switches to three-valued mode.
  - test target note: `test-semantic-filter-missing-unknown`.
- Project / expression operators:
  - computed expressions with missing inputs produce missing outputs by default (strict mode).
  - explicit numeric kernels may choose explicit mask-preserving promotion rules for scalar arithmetic.
  - test target note: `test-semantic-project-missing-expression`.
- Join operators:
  - missing values in join keys do not match any non-missing key and never match each other unless key comparator explicitly opts in.
  - row-presence semantics remain join-type dependent (`inner`, `left`, `right`, `full`), with missing-key treatment applied before join-type semantics.
  - test target note: `test-semantic-join-missing-key`.
- Group-by operators:
  - `NULL`/missing in group keys are assigned to a dedicated missing group bucket per key tuple.
  - within aggregates, missing accumulation inputs are ignored by default for idempotent reduction families (e.g., `sum`, `mean`, `count`) unless aggregate declares `count-missing=true`.
  - test target note: `test-semantic-groupby-missing-key-grouping`.
- Order-by / window operators:
  - sort keys containing missing values are treated deterministically using explicit null positioning policy:
    - default nulls last for ascending order,
    - default nulls first for descending order.
  - window boundaries reference the ordered materialized row stream after null placement.
  - test target note: `test-semantic-order-by-missing-placement`.
- Limit / top-k operators:
  - limit is applied after ordering/filtering and does not alter missingness payload.
  - test target note: `test-semantic-limit-missing-preserves-shape`.
- Reductions / analytic operators:
  - `count` counts missing-aware and all rows depending on mode (`count` vs `count-non-missing`).
  - reduction initial accumulator:
    - strict: first non-missing input seeds accumulator when possible,
    - permissive: first input seeds even if missing and carries missing forward.
  - test target note: `test-semantic-reduction-missing-accumulator`.
- Boolean logic operators:
  - tri-state policy is explicit per function; base mode treats missing as `false` for predicate-like operators.
  - operators explicitly opting into unknown-propagating mode may emit missing outputs instead.
  - test target note: `test-semantic-logic-missing-tristate`.

- Missing propagation defaults are documented as strict baseline unless a compatibility mode explicitly relaxes behavior.

### 6.4.4 Duplicate and grouping edge cases
- Grouping with an invalid key should error unless a documented fallback mode is configured.
- Invalid grouping keys must not silently coerce; emit a structured error.
- Empty groups are not emitted unless `emit-empty-groups` is requested.

### 6.4.5 Window and ordering semantics
- Ordered operations must have a deterministic total order input; ties must be resolved with a second deterministic key.
- Default tie-break key is `(source_row_index, deterministic_operator_instance_id)` unless overridden.
- Window boundaries are evaluated on finalized order only; partial reordering is disallowed.

### 6.4.6 Determinism and reproducibility
- Deterministic mode:
  - fixes random stream seeds at the API boundary,
  - requires deterministic reduction tree shape when possible,
  - enables explicit stable ordering in all user-visible operations.
- Non-deterministic mode may use faster algorithms but must preserve documented accuracy and error bounds.

### 6.4.7 Type promotion and error policy
- Integer arithmetic promotes to float only when an operator requires fraction semantics.
- Mixed numeric inputs follow a documented promotion lattice:
  - int32 + int32 -> int32
  - int32 + int64 -> int64
  - int32/int64 + float32 -> float32
  - float32 + float64 -> float64
- Invalid window specs, unsupported reductions, and unsupported dtypes are hard errors.
- Null/nullable inputs must surface a structured error in strict mode or follow documented fallback in compatibility mode.

### 6.4.8 Traceability and diagnostics
- Every failing plan step must include operation name, input relation/tensor identifiers, and argument snapshot.
- Error messages must be stable and keyed by operator code path.

### 6.4.9 Keyword compatibility policy
- Public API functions accept a fixed, documented keyword argument set per function.
- Unrecognized keyword arguments must fail immediately with a structured error containing:
  - operation name,
  - unrecognized key(s),
  - allowed key set.
- Unknown positional arity combinations are rejected, not coerced.
- Keyword aliases are only permitted when explicitly listed as part of the same backward-compatibility policy document for that API family.
- Compatibility mode is opt-in and versioned: it must be declared per call/site and may only accept deprecation-backed aliases for that exact major version.

### 6.4.10 Semantic rule test mappings
- `6.4.1`: `test-m1-lanes`
- `6.4.2`: `test-m1-set-behavior`
- `6.4.3`: `test-m1-missing-invalid-values`
- `6.4.3a`: `test-m1-missing-by-operator`
- `6.4.4`: `test-m1-grouping-edge-cases`
- `6.4.5`: `test-m1-window-ordering`
- `6.4.6`: `test-m1-determinism-reproducibility`
- `6.4.7`: `test-m1-type-promotion`
- `6.4.8`: `test-m1-traceability-errors`
- `6.4.9`: `test-m1-keyword-compatibility`

## 7. Phased Delivery Plan

## Phase 0: Foundation hardening (1-2 weeks)
- split existing `lib/torch.omni` into `core` modules
- add uniform error model and argument validation helpers
- add handle lifecycle tests (create/free/reuse failure modes)
- add Docker dev profile with 30% CPU cap

Deliverables:
- modularized API
- baseline test harness and CI jobs

## Phase 1: Trainable core (2-3 weeks)
- autograd wrappers (`requires-grad`, `backward`, `detach`, `no-grad`)
- `nn/module`, `nn/parameter`, `nn/sequential`
- layers: linear, embedding, dropout, activations
- losses: MSE, BCE, cross-entropy
- optimizers: SGD, Adam, AdamW

Deliverables:
- MNIST-like classifier training end-to-end

## Phase 2: Data and production training loop (2-3 weeks)
- dataset/dataloader/sampler/transforms
- callbacks, checkpointing, metrics history
- scheduler support (warmup + cosine/step)

Deliverables:
- reproducible training pipelines with resume-from-checkpoint

## Phase 3: Model zoo and inference UX (2 weeks)
- model zoo: MLP, 1D CNN, small transformer encoder
- inference API (`predict`, batched inference, export/import state)
- examples and docs for common tasks

Deliverables:
- ready-to-run examples with measured metrics

## Phase 4: Scaling and acceleration (optional)
- mixed precision (where backend permits)
- CUDA device placement policy
- data-parallel/multi-process strategy

Deliverables:
- larger model training support and perf dashboards

## 8. Testing Strategy

- Unit tests: tensor ops, modules, optimizer steps, serialization.
- Property tests: shape-preserving ops, gradient sanity checks.
- Golden tests: expected outputs for known seeds.
- Regression tests: previously fixed bugs in tensor shape/autograd/memory.
- Integration tests: full train/eval/checkpoint lifecycle.

Suggested test directories:

```
tests/
  core/
  autograd/
  nn/
  optim/
  data/
  io/
  integration/
```

## 9. First Backlog (Concrete Next Items)
1. Create module split (`lib/core/*`) and move non-FFI helpers out of `lib/torch.omni`.
2. Add `autograd` FFI bindings in `lib/ffi/torch.omni` and wrappers in `lib/autograd/grad.omni`.
3. Implement `nn/module`, `nn/parameter`, `nn/sequential`.
4. Implement `optim/sgd` and `optim/adamw` with parameter iteration API.
5. Add `train/loop.omni` with minimal `fit` and callback hooks.
6. Add Docker dev profile enforcing resource caps (CPU <= 30%).
7. Add CI matrix: lint, unit tests, integration mini-train job.

## 10. Definition of Done for v1
- A user can define a model, train it, checkpoint it, and run inference entirely in Omni.
- API has stable module boundaries and documented behavior.
- Test suite covers core functionality and memory/resource safety.
- Docker-based workflows are default and resource-capped.
