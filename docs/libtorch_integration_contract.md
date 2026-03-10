## Libtorch Integration Contract

## Purpose

This document defines the explicit contract between Omni Torch and the libtorch backend
for the currently implemented surface (statistics and cleanup-oriented core ops).

## 1) Backend abstraction boundary

- All public tensor execution must pass through `lib/torch.omni`.
- `lib/torch.omni` owns FFI calls and is the single boundary for:
  - tensor creation and shape allocation
  - dtype conversion
  - numeric and analytic tensor ops
  - missing/non-finite mask creation
  - statistic entrypoints and reduction paths
- C++ shim ownership lives in `csrc/torch_shim.cpp` with declarations in `csrc/torch_shim.h` and
  FFI entrypoints in `lib/ffi/torch.omni`.
- Executor context/dispatch is expected to treat these functions as opaque backend operations.

## 2) Dtype mapping

Omni Torch exposes the following dtype constants in `lib/torch.omni`:

| Omni constant | Value | C++ `at::ScalarType` |
| --- | ---: | --- |
| `torch/float32` | `6` | `at::kFloat` |
| `torch/float64` | `7` | `at::kDouble` |
| `torch/int32`   | `3` | `at::kInt` |
| `torch/int64`   | `4` | `at::kLong` |

Mapping is enforced by `omni_torch_to_dtype` in the C++ shim.

### Supported conversion policy

- Explicit conversion is supported via `tensor/to-dtype`.
- Statistical helpers in this release upcast to `float64` internally when needed for numerically stable accumulation.
- Unsupported input dtypes should surface as a backend error and fail with a stable Omni diagnostic message.

## 3) Device mapping and support policy

- Current support target is **CPU-first**.
- `at::kCUDA` pathways are intentionally not exposed in this milestone.
- Any attempt to target non-CPU placement must return a clear backend diagnostic rather than silently reinterpreting.

## 4) Layout and contiguity semantics

- Contiguity is currently exposed via `tensor/contiguous?`.
- Most operations pass through to libtorch with natural libtorch behavior:
  - contiguous inputs run directly,
  - non-contiguous inputs are accepted by libtorch kernels that can operate on them,
  - kernels that require contiguous inputs are allowed to materialize implicit copies inside the backend.
- Public API does not currently require callers to pre-normalize layout.

## 5) Missing and non-finite handling

- Masks are first-class via:
  - `tensor/is-not-a-number`
  - `tensor/is-infinite`
  - `tensor/is-finite`
- Deterministic replacement in-place uses `tensor/fill-missing-inplace`.
- Statistics modules for missing values use the same mask/replacement policy and rely on `stats/missing/*` wrappers.
- Invalid non-finite replacement values are delegated to libtorch nan/inf handling and must fail loudly if unsupported.

## 6) Diagnostics and failure translation

- The C++ shim stores backend errors in a thread-local error slot in `g_last_error`.
- On exception, shim functions return a null handle (or complete without value for in-place/void calls) and set an error message.
- `lib/torch.omni` reads this value through `omni-torch-last-error` and converts it into stable Omni errors in `_wrap` paths.
- For void operations (`tensor/set-num-threads`, in-place updates), callers should check post-call backend error state and re-raise as an Omni error.

## 7) Thread controls

- Runtime thread controls are surfaced with `tensor/set-num-threads`.
- This is wired to `at::set_num_threads` in `csrc/torch_shim.cpp` and rejects non-positive values.
- Future executor contexts should call this once during initialization and treat failures as startup validation errors.
- `tensor/get-num-threads` is available for control plane diagnostics and should mirror the runtime value used by libtorch.

## 7b) Memory controls and reporting

- `tensor/set-memory-limit-bytes` stores an execution ceiling for the shim-managed tensor lifetime accounting.
- `tensor/memory-report` returns a tuple-like list of:
  - configured memory limit (bytes),
  - currently tracked allocated bytes (bytes),
  - peak observed tracked bytes (bytes).
- Memory accounting is approximation-based and tracks bytes from shim-created tensor handles; it does not include:
  - allocator fragmentation,
  - untracked external storage sharing,
  - and framework-internal caches.
- This reporting is expected to be used for executor-side guardrails and telemetry, not strict allocator enforcement semantics.

## 8) Initial operator coverage for initial stats/cleanup surface

Initial coverage is complete in the current codebase for the first statistics surface and missing-value handling ops:

- `stats/describe/*`
- `stats/robust/*`
- `stats/distribution/*`
- `stats/association/*`
- `stats/missing/*`
- Utility cleanup-oriented operations already supported: `tensor/is-*` masks and `tensor/fill-missing-inplace`.

## 9) Unsupported/unknown behaviors

- No fallback path for unsupported tensor devices is defined yet; unsupported operations must fail with explicit backend diagnostics.
- Shim-level memory-pressure controls and reporting are implemented for byte-tracked allocations, but full allocator-level accounting and hard enforcement remain follow-on items.
- `libtorch` feature coverage should be extended with conformance tests before enabling any broader feature claims.
