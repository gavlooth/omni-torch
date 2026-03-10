#include "torch_shim.h"
#include <torch/torch.h>
#include <ATen/ops/_unique2.h>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <tuple>
#include <limits>
#include <cmath>

// Thread-local error buffer
static thread_local std::string g_last_error;
static thread_local bool g_has_error = false;
static std::atomic<int64_t> g_memory_limit_bytes{0};
static std::atomic<int64_t> g_current_memory_bytes{0};
static std::atomic<int64_t> g_peak_memory_bytes{0};

// Macro: wrap C++ exceptions into error string
#define CATCH_ERR(default_ret) \
    catch (const std::exception& e) { \
        g_last_error = e.what(); \
        g_has_error = true; \
        return default_ret; \
    } catch (...) { \
        g_last_error = "unknown error"; \
        g_has_error = true; \
        return default_ret; \
    }

#define CLEAR_ERR() g_has_error = false

static inline at::Tensor* to_tensor(TensorHandle h) {
    return reinterpret_cast<at::Tensor*>(h);
}

static inline int64_t tensor_nbytes(const at::Tensor& t) {
    return static_cast<int64_t>(t.nbytes());
}

static inline void register_tensor_allocation(int64_t bytes) {
    if (bytes <= 0) {
        return;
    }

    int64_t limit = g_memory_limit_bytes.load(std::memory_order_relaxed);
    while (true) {
        int64_t current = g_current_memory_bytes.load(std::memory_order_relaxed);
        int64_t projected = current + bytes;
        if (limit > 0 && projected > limit) {
            throw std::runtime_error("memory limit exceeded for libtorch backend");
        }
        if (g_current_memory_bytes.compare_exchange_weak(current, projected, std::memory_order_relaxed)) {
            int64_t peak = g_peak_memory_bytes.load(std::memory_order_relaxed);
            while (projected > peak) {
                if (g_peak_memory_bytes.compare_exchange_weak(peak, projected, std::memory_order_relaxed)) {
                    break;
                }
            }
            return;
        }
    }
}

static inline void unregister_tensor_allocation(int64_t bytes) {
    if (bytes <= 0) {
        return;
    }
    int64_t old = g_current_memory_bytes.fetch_sub(bytes, std::memory_order_relaxed);
    if (old <= bytes) {
        g_current_memory_bytes.store(0, std::memory_order_relaxed);
    }
}

static inline TensorHandle from_tensor(at::Tensor t) {
    register_tensor_allocation(tensor_nbytes(t));
    auto* p = new at::Tensor(std::move(t));
    return reinterpret_cast<TensorHandle>(p);
}

// === Memory and runtime controls ===

extern "C" int64_t omni_torch_get_num_threads(void) {
    CLEAR_ERR();
    try { return at::get_num_threads(); } CATCH_ERR(-1)
}

extern "C" void omni_torch_set_memory_limit_bytes(int64_t bytes) {
    CLEAR_ERR();
    try {
        if (bytes < 0) {
            throw std::runtime_error("memory limit bytes must be >= 0");
        }
        g_memory_limit_bytes.store(bytes, std::memory_order_relaxed);
    } CATCH_ERR()
}

extern "C" int64_t omni_torch_get_memory_limit_bytes(void) {
    CLEAR_ERR();
    return g_memory_limit_bytes.load(std::memory_order_relaxed);
}

extern "C" int64_t omni_torch_get_memory_allocated_bytes(void) {
    CLEAR_ERR();
    return g_current_memory_bytes.load(std::memory_order_relaxed);
}

extern "C" int64_t omni_torch_get_memory_peak_bytes(void) {
    CLEAR_ERR();
    return g_peak_memory_bytes.load(std::memory_order_relaxed);
}

static inline at::ScalarType to_dtype(int64_t dtype) {
    switch (dtype) {
        case OMNI_FLOAT32: return at::kFloat;
        case OMNI_FLOAT64: return at::kDouble;
        case OMNI_INT32:   return at::kInt;
        case OMNI_INT64:   return at::kLong;
        default:           return at::kFloat;
    }
}

// === Error Handling ===

extern "C" const char* omni_torch_last_error(void) {
    if (g_has_error) {
        g_has_error = false;
        return g_last_error.c_str();
    }
    return nullptr;
}

// === Tensor Creation ===

extern "C" TensorHandle omni_torch_zeros_1d(int64_t d0, int64_t dtype) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::zeros({d0}, to_dtype(dtype)));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_zeros_2d(int64_t d0, int64_t d1, int64_t dtype) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::zeros({d0, d1}, to_dtype(dtype)));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_zeros_3d(int64_t d0, int64_t d1, int64_t d2, int64_t dtype) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::zeros({d0, d1, d2}, to_dtype(dtype)));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_ones_1d(int64_t d0, int64_t dtype) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::ones({d0}, to_dtype(dtype)));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_ones_2d(int64_t d0, int64_t d1, int64_t dtype) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::ones({d0, d1}, to_dtype(dtype)));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_ones_3d(int64_t d0, int64_t d1, int64_t d2, int64_t dtype) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::ones({d0, d1, d2}, to_dtype(dtype)));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_rand_1d(int64_t d0) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::rand({d0}));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_rand_2d(int64_t d0, int64_t d1) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::rand({d0, d1}));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_rand_3d(int64_t d0, int64_t d1, int64_t d2) {
    CLEAR_ERR();
    try { return from_tensor(torch::rand({d0, d1, d2})); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_randn_3d(int64_t d0, int64_t d1, int64_t d2) {
    CLEAR_ERR();
    try { return from_tensor(torch::randn({d0, d1, d2})); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_randn_1d(int64_t d0) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::randn({d0}));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_randn_2d(int64_t d0, int64_t d1) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::randn({d0, d1}));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_arange(int64_t start, int64_t end, int64_t step) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::arange(start, end, step));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_linspace(int64_t n) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::linspace(0.0, 1.0, n));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_eye(int64_t n) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::eye(n));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_from_blob_f32(int64_t data_ptr, int64_t numel) {
    CLEAR_ERR();
    try {
        float* p = reinterpret_cast<float*>(data_ptr);
        // Clone so the tensor owns its memory
        return from_tensor(torch::from_blob(p, {numel}, at::kFloat).clone());
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_from_blob_f64(int64_t data_ptr, int64_t numel) {
    CLEAR_ERR();
    try {
        double* p = reinterpret_cast<double*>(data_ptr);
        return from_tensor(torch::from_blob(p, {numel}, at::kDouble).clone());
    } CATCH_ERR(0)
}

// === Tensor Info ===

extern "C" int64_t omni_torch_dim(TensorHandle t) {
    CLEAR_ERR();
    try { return to_tensor(t)->dim(); } CATCH_ERR(-1)
}

extern "C" int64_t omni_torch_numel(TensorHandle t) {
    CLEAR_ERR();
    try { return to_tensor(t)->numel(); } CATCH_ERR(-1)
}

extern "C" int64_t omni_torch_shape(TensorHandle t, int64_t dim_idx) {
    CLEAR_ERR();
    try { return to_tensor(t)->size(dim_idx); } CATCH_ERR(-1)
}

extern "C" int64_t omni_torch_dtype(TensorHandle t) {
    CLEAR_ERR();
    try {
        auto st = to_tensor(t)->scalar_type();
        switch (st) {
            case at::kFloat:  return OMNI_FLOAT32;
            case at::kDouble: return OMNI_FLOAT64;
            case at::kInt:    return OMNI_INT32;
            case at::kLong:   return OMNI_INT64;
            default:          return static_cast<int64_t>(st);
        }
    } CATCH_ERR(-1)
}

extern "C" int64_t omni_torch_is_contiguous(TensorHandle t) {
    CLEAR_ERR();
    try { return to_tensor(t)->is_contiguous() ? 1 : 0; } CATCH_ERR(0)
}

extern "C" int64_t omni_torch_data_ptr(TensorHandle t) {
    CLEAR_ERR();
    try { return reinterpret_cast<int64_t>(to_tensor(t)->data_ptr()); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_is_not_a_number(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::isnan(*to_tensor(t))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_is_infinite(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::isinf(*to_tensor(t))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_is_finite(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::isfinite(*to_tensor(t))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_nan_to_num(TensorHandle t, double nan, double posinf, double neginf) {
    CLEAR_ERR();
    try { return from_tensor(torch::nan_to_num(*to_tensor(t), nan, posinf, neginf)); } CATCH_ERR(0)
}

static inline at::Tensor ensure_double_1d(TensorHandle t) {
    auto tensor = *to_tensor(t);
    return tensor.reshape({-1}).to(at::kDouble);
}

static inline at::Tensor ensure_mean_tensor(const at::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case at::kBool:
        case at::kByte:
        case at::kChar:
        case at::kShort:
        case at::kInt:
        case at::kLong:
            return tensor.to(at::kDouble);
        default:
            return tensor;
    }
}

static inline at::Tensor ensure_float_math_tensor(const at::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case at::kBool:
        case at::kByte:
        case at::kChar:
        case at::kShort:
        case at::kInt:
        case at::kLong:
            return tensor.to(at::kDouble);
        default:
            return tensor;
    }
}

static inline int64_t tensor_scalar_to_i64(const at::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case at::kFloat:
            return static_cast<int64_t>(tensor.item<float>());
        case at::kDouble:
            return static_cast<int64_t>(tensor.item<double>());
        case at::kInt:
            return static_cast<int64_t>(tensor.item<int32_t>());
        case at::kLong:
            return tensor.item<int64_t>();
        case at::kBool:
            return tensor.item<bool>() ? 1 : 0;
        default:
            return static_cast<int64_t>(tensor.to(at::kLong).item<int64_t>());
    }
}

static inline std::pair<at::Tensor, at::Tensor> ensure_binary_linalg_tensors(
    const at::Tensor& a, const at::Tensor& b) {
    auto left = ensure_float_math_tensor(a);
    auto right = ensure_float_math_tensor(b);
    if (left.scalar_type() == right.scalar_type()) {
        return {left, right};
    }
    return {left.to(at::kDouble), right.to(at::kDouble)};
}

extern "C" TensorHandle omni_torch_median(TensorHandle t) {
    CLEAR_ERR();
    try {
        auto x = ensure_double_1d(t);
        auto n = x.numel();
        if (n == 0) {
            throw std::runtime_error("median requires non-empty tensor");
        }
        auto sorted = std::get<0>(torch::sort(x, 0));
        auto opts = torch::TensorOptions().dtype(torch::kLong);
        int64_t mid = n / 2;
        if ((n % 2) == 1) {
            auto idx = torch::tensor({mid}, opts);
            return from_tensor(sorted.index_select(0, idx).reshape({}));
        }
        auto idx = torch::tensor({mid - 1, mid}, opts);
        auto pair = sorted.index_select(0, idx);
        auto lo = pair.narrow(0, 0, 1).reshape({});
        auto hi = pair.narrow(0, 1, 1).reshape({});
        return from_tensor((lo + hi) / 2.0);
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_median_dim(TensorHandle t, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(std::get<0>(to_tensor(t)->median(dim))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_quantile(TensorHandle t, double q) {
    CLEAR_ERR();
    try {
        if (q < 0.0 || q > 1.0) {
            throw std::runtime_error("quantile requires q in [0, 1]");
        }
        auto x = ensure_double_1d(t);
        auto n = x.numel();
        if (n == 0) {
            throw std::runtime_error("quantile requires non-empty tensor");
        }
        auto sorted = std::get<0>(torch::sort(x, 0));
        double pos = (static_cast<double>(n - 1)) * q;
        auto lo_i = static_cast<int64_t>(std::floor(pos));
        auto hi_i = static_cast<int64_t>(std::ceil(pos));
        auto opts = torch::TensorOptions().dtype(torch::kLong);
        auto alpha = pos - static_cast<double>(lo_i);
        if (lo_i == hi_i) {
            auto idx = torch::tensor({lo_i}, opts);
            return from_tensor(sorted.index_select(0, idx).reshape({}));
        }
        auto idx = torch::tensor({lo_i, hi_i}, opts);
        auto pair = sorted.index_select(0, idx);
        auto lo = pair.narrow(0, 0, 1).reshape({});
        auto hi = pair.narrow(0, 1, 1).reshape({});
        return from_tensor(lo + (hi - lo) * alpha);
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_quantile_dim(TensorHandle t, double q, int64_t dim) {
    CLEAR_ERR();
    try {
        if (q < 0.0 || q > 1.0) {
            throw std::runtime_error("quantile requires q in [0, 1]");
        }
        auto x = *to_tensor(t);
        if (x.dim() == 0) {
            throw std::runtime_error("quantile-dim requires a tensor with at least 1 dimension");
        }
        if (dim < 0) {
            dim += x.dim();
        }
        if (dim < 0 || dim >= x.dim()) {
            throw std::runtime_error("quantile-dim requires dim in tensor range");
        }
        auto n = x.size(dim);
        if (n <= 0) {
            throw std::runtime_error("quantile-dim requires non-empty target dimension");
        }
        auto sorted = std::get<0>(x.to(at::kDouble).sort(dim));
        double pos = (static_cast<double>(n - 1)) * q;
        auto lo_i = static_cast<int64_t>(std::floor(pos));
        auto hi_i = static_cast<int64_t>(std::ceil(pos));
        auto alpha = pos - static_cast<double>(lo_i);
        if (lo_i == hi_i) {
            return from_tensor(sorted.select(dim, lo_i));
        }
        auto lo = sorted.select(dim, lo_i);
        auto hi = sorted.select(dim, hi_i);
        return from_tensor(lo + (hi - lo) * alpha);
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_covariance(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try {
        auto xa = ensure_double_1d(a);
        auto xb = ensure_double_1d(b);
        if (xa.numel() != xb.numel()) {
            throw std::runtime_error("covariance expects two tensors with equal numel");
        }
        auto n = xa.numel();
        if (n < 2) {
            throw std::runtime_error("covariance requires at least 2 elements");
        }
        auto centered_a = xa - xa.mean();
        auto centered_b = xb - xb.mean();
        auto denom = static_cast<double>(n - 1);
        return from_tensor((centered_a * centered_b).sum() / denom);
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_correlation(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try {
        auto xa = ensure_double_1d(a);
        auto xb = ensure_double_1d(b);
        if (xa.numel() != xb.numel()) {
            throw std::runtime_error("correlation expects two tensors with equal numel");
        }
        auto n = xa.numel();
        if (n < 2) {
            throw std::runtime_error("correlation requires at least 2 elements");
        }
        auto centered_a = xa - xa.mean();
        auto centered_b = xb - xb.mean();
        auto denom = static_cast<double>(n - 1);
        auto cov = (centered_a * centered_b).sum() / denom;
        auto var_a = (centered_a * centered_a).sum() / denom;
        auto var_b = (centered_b * centered_b).sum() / denom;
        auto den = torch::sqrt(var_a * var_b);
        if (den.item<double>() == 0.0) {
            return from_tensor(at::full({}, std::numeric_limits<double>::quiet_NaN(), at::TensorOptions().dtype(at::kDouble)));
        }
        return from_tensor(cov / den);
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_histogram(TensorHandle t, int64_t bins, double min, double max) {
    CLEAR_ERR();
    try {
        if (bins <= 0) {
            throw std::runtime_error("histogram requires bins > 0");
        }
        auto x = to_tensor(t)->reshape({-1}).to(at::kDouble);
        if (x.numel() == 0) {
            return from_tensor(torch::zeros({bins}, x.options()));
        }
        if (std::isnan(min) || std::isnan(max) || max < min) {
            throw std::runtime_error("histogram requires finite range with max >= min");
        }
        return from_tensor(torch::histc(x, bins, min, max));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_unique_values_with_counts(TensorHandle t) {
    CLEAR_ERR();
    try {
        auto x = to_tensor(t)->reshape({-1});
        if (x.numel() == 0) {
            return from_tensor(torch::empty({0, 0}, torch::TensorOptions().dtype(at::kDouble)));
        }
        auto tuple_vals = at::_unique2(x, true, false, true);
        auto values = std::get<0>(tuple_vals);
        auto counts = std::get<2>(tuple_vals);
        return from_tensor(torch::stack({values.to(at::kDouble), counts.to(at::kDouble)}, 0));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_count_distinct(TensorHandle t) {
    CLEAR_ERR();
    try {
        auto x = to_tensor(t)->reshape({-1});
        if (x.numel() == 0) {
            return from_tensor(torch::tensor(0, torch::TensorOptions().dtype(torch::kInt64)));
        }
        auto unique = std::get<0>(at::_unique2(x, true, false, false));
        auto count = unique.numel();
        return from_tensor(torch::tensor(static_cast<int64_t>(count), torch::TensorOptions().dtype(torch::kInt64)));
    } CATCH_ERR(0)
}

// === Element Access ===

extern "C" double omni_torch_item_f64(TensorHandle t) {
    CLEAR_ERR();
    try { return to_tensor(t)->item<double>(); } CATCH_ERR(0.0)
}

extern "C" int64_t omni_torch_item_i64(TensorHandle t) {
    CLEAR_ERR();
    try { return tensor_scalar_to_i64(*to_tensor(t)); } CATCH_ERR(0)
}

extern "C" double omni_torch_get_1d_f64(TensorHandle t, int64_t i) {
    CLEAR_ERR();
    try { return to_tensor(t)->index({i}).item<double>(); } CATCH_ERR(0.0)
}

extern "C" double omni_torch_get_2d_f64(TensorHandle t, int64_t i, int64_t j) {
    CLEAR_ERR();
    try { return to_tensor(t)->index({i, j}).item<double>(); } CATCH_ERR(0.0)
}

extern "C" int64_t omni_torch_get_1d_i64(TensorHandle t, int64_t i) {
    CLEAR_ERR();
    try { return tensor_scalar_to_i64(to_tensor(t)->index({i})); } CATCH_ERR(0)
}

extern "C" int64_t omni_torch_get_2d_i64(TensorHandle t, int64_t i, int64_t j) {
    CLEAR_ERR();
    try { return tensor_scalar_to_i64(to_tensor(t)->index({i, j})); } CATCH_ERR(0)
}

// === Arithmetic ===

#define BINARY_OP(name, op) \
extern "C" TensorHandle omni_torch_##name(TensorHandle a, TensorHandle b) { \
    CLEAR_ERR(); \
    try { return from_tensor(*to_tensor(a) op *to_tensor(b)); } CATCH_ERR(0) \
}

BINARY_OP(add, +)
BINARY_OP(sub, -)
BINARY_OP(mul, *)
BINARY_OP(div, /)

extern "C" TensorHandle omni_torch_neg(TensorHandle a) {
    CLEAR_ERR();
    try { return from_tensor(-*to_tensor(a)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_abs(TensorHandle a) {
    CLEAR_ERR();
    try { return from_tensor(torch::abs(*to_tensor(a))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_add_scalar_i(TensorHandle a, int64_t s) {
    CLEAR_ERR();
    try { return from_tensor(*to_tensor(a) + s); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_sub_scalar_i(TensorHandle a, int64_t s) {
    CLEAR_ERR();
    try { return from_tensor(*to_tensor(a) - s); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_mul_scalar_i(TensorHandle a, int64_t s) {
    CLEAR_ERR();
    try { return from_tensor(*to_tensor(a) * s); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_div_scalar_i(TensorHandle a, int64_t s) {
    CLEAR_ERR();
    try { return from_tensor(*to_tensor(a) / s); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_add_scalar_f(TensorHandle a, double s) {
    CLEAR_ERR();
    try { return from_tensor(*to_tensor(a) + s); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_sub_scalar_f(TensorHandle a, double s) {
    CLEAR_ERR();
    try { return from_tensor(*to_tensor(a) - s); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_mul_scalar_f(TensorHandle a, double s) {
    CLEAR_ERR();
    try { return from_tensor(*to_tensor(a) * s); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_div_scalar_f(TensorHandle a, double s) {
    CLEAR_ERR();
    try { return from_tensor(*to_tensor(a) / s); } CATCH_ERR(0)
}

// === Linear Algebra ===

extern "C" TensorHandle omni_torch_matmul(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try {
        auto tensors = ensure_binary_linalg_tensors(*to_tensor(a), *to_tensor(b));
        return from_tensor(torch::matmul(tensors.first, tensors.second));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_mm(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try {
        auto tensors = ensure_binary_linalg_tensors(*to_tensor(a), *to_tensor(b));
        return from_tensor(torch::mm(tensors.first, tensors.second));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_mv(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try {
        auto tensors = ensure_binary_linalg_tensors(*to_tensor(a), *to_tensor(b));
        return from_tensor(torch::mv(tensors.first, tensors.second));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_dot(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try {
        auto tensors = ensure_binary_linalg_tensors(*to_tensor(a), *to_tensor(b));
        return from_tensor(torch::dot(tensors.first, tensors.second));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_transpose(TensorHandle a, int64_t d0, int64_t d1) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->transpose(d0, d1).contiguous()); } CATCH_ERR(0)
}

// === Reductions ===

extern "C" TensorHandle omni_torch_sum(TensorHandle a) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->sum()); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_mean(TensorHandle a) {
    CLEAR_ERR();
    try {
        auto x = ensure_mean_tensor(*to_tensor(a));
        return from_tensor(x.mean());
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_min(TensorHandle a) {
    CLEAR_ERR();
    try { return from_tensor(std::get<0>(to_tensor(a)->min(0))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_max(TensorHandle a) {
    CLEAR_ERR();
    try { return from_tensor(std::get<0>(to_tensor(a)->max(0))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_sum_dim(TensorHandle a, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->sum(dim)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_mean_dim(TensorHandle a, int64_t dim) {
    CLEAR_ERR();
    try {
        auto x = ensure_mean_tensor(*to_tensor(a));
        return from_tensor(x.mean(dim));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_min_dim(TensorHandle a, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(std::get<0>(to_tensor(a)->min(dim))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_max_dim(TensorHandle a, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(std::get<0>(to_tensor(a)->max(dim))); } CATCH_ERR(0)
}

// === Shape Operations ===

extern "C" TensorHandle omni_torch_reshape_1d(TensorHandle a, int64_t d0) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->reshape({d0})); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_reshape_2d(TensorHandle a, int64_t d0, int64_t d1) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->reshape({d0, d1})); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_reshape_3d(TensorHandle a, int64_t d0, int64_t d1, int64_t d2) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->reshape({d0, d1, d2})); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_squeeze(TensorHandle a) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->squeeze()); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_unsqueeze(TensorHandle a, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->unsqueeze(dim)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_flatten(TensorHandle a) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->flatten()); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_cat_2(TensorHandle a, TensorHandle b, int64_t dim) {
    CLEAR_ERR();
    try {
        std::vector<at::Tensor> v = {*to_tensor(a), *to_tensor(b)};
        return from_tensor(torch::cat(v, dim));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_stack_2(TensorHandle a, TensorHandle b, int64_t dim) {
    CLEAR_ERR();
    try {
        std::vector<at::Tensor> v = {*to_tensor(a), *to_tensor(b)};
        return from_tensor(torch::stack(v, dim));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_select(TensorHandle a, int64_t dim, int64_t index) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->select(dim, index).contiguous()); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_slice(TensorHandle a, int64_t dim, int64_t start, int64_t end) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->slice(dim, start, end)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_index_select(TensorHandle a, int64_t dim, TensorHandle idx) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->index_select(dim, *to_tensor(idx))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_clone(TensorHandle a) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->clone()); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_contiguous(TensorHandle a) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->contiguous()); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_to_dtype(TensorHandle a, int64_t dtype) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->to(to_dtype(dtype))); } CATCH_ERR(0)
}

// === Math ===

#define UNARY_MATH(name, fn) \
extern "C" TensorHandle omni_torch_##name(TensorHandle a) { \
    CLEAR_ERR(); \
    try { return from_tensor(torch::fn(ensure_float_math_tensor(*to_tensor(a)))); } CATCH_ERR(0) \
}

UNARY_MATH(exp, exp)
UNARY_MATH(log, log)
UNARY_MATH(sqrt, sqrt)
UNARY_MATH(sin, sin)
UNARY_MATH(cos, cos)
UNARY_MATH(tanh, tanh)
UNARY_MATH(sigmoid, sigmoid)

extern "C" TensorHandle omni_torch_pow_scalar(TensorHandle a, int64_t exp) {
    CLEAR_ERR();
    try { return from_tensor(torch::pow(*to_tensor(a), exp)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_pow_scalar_f(TensorHandle a, double exp) {
    CLEAR_ERR();
    try { return from_tensor(torch::pow(*to_tensor(a), exp)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_clamp(TensorHandle a, double min_val, double max_val) {
    CLEAR_ERR();
    try { return from_tensor(torch::clamp(*to_tensor(a), min_val, max_val)); } CATCH_ERR(0)
}

// === Comparison ===

#define COMPARE_OP(name, op) \
extern "C" TensorHandle omni_torch_##name(TensorHandle a, TensorHandle b) { \
    CLEAR_ERR(); \
    try { return from_tensor(*to_tensor(a) op *to_tensor(b)); } CATCH_ERR(0) \
}

COMPARE_OP(eq, ==)
COMPARE_OP(lt, <)
COMPARE_OP(gt, >)
COMPARE_OP(le, <=)
COMPARE_OP(ge, >=)

// === NN ===

extern "C" TensorHandle omni_torch_relu(TensorHandle a) {
    CLEAR_ERR();
    try { return from_tensor(torch::relu(*to_tensor(a))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_softmax(TensorHandle a, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(torch::softmax(ensure_float_math_tensor(*to_tensor(a)), dim)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_log_softmax(TensorHandle a, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(torch::log_softmax(ensure_float_math_tensor(*to_tensor(a)), dim)); } CATCH_ERR(0)
}

// === In-Place ===

extern "C" void omni_torch_fill_i64(TensorHandle a, int64_t val) {
    CLEAR_ERR();
    try { to_tensor(a)->fill_(val); } catch (...) {}
}

// === Memory ===

extern "C" void omni_torch_free(TensorHandle t) {
    if (t) {
        int64_t bytes = tensor_nbytes(*to_tensor(t));
        delete to_tensor(t);
        unregister_tensor_allocation(std::max<int64_t>(0, bytes));
    }
}

// === Display ===

extern "C" void omni_torch_print(TensorHandle t) {
    CLEAR_ERR();
    try { std::cout << *to_tensor(t) << std::endl; } catch (...) {}
}

extern "C" int64_t omni_torch_to_string(TensorHandle t, int64_t buf_ptr, int64_t buf_len) {
    CLEAR_ERR();
    try {
        std::ostringstream oss;
        oss << *to_tensor(t);
        std::string s = oss.str();
        char* buf = reinterpret_cast<char*>(buf_ptr);
        int64_t copy_len = std::min((int64_t)s.size(), buf_len - 1);
        memcpy(buf, s.c_str(), copy_len);
        buf[copy_len] = '\0';
        return copy_len;
    } CATCH_ERR(0)
}

// ============================================================
// FFT operations (for Wave-PDE spectral Laplacian)
// ============================================================

extern "C" TensorHandle omni_torch_fft_rfft(TensorHandle t, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(torch::fft::rfft(*to_tensor(t), {}, dim)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_fft_irfft(TensorHandle t, int64_t n, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(torch::fft::irfft(*to_tensor(t), n, dim)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_fft_fftfreq(int64_t n) {
    CLEAR_ERR();
    try { return from_tensor(torch::fft::fftfreq(n, 1.0 / n)); } CATCH_ERR(0)
}

// ============================================================
// Like-constructors
// ============================================================

extern "C" TensorHandle omni_torch_zeros_like(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::zeros_like(*to_tensor(t))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_ones_like(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::ones_like(*to_tensor(t))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_randn_like(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::randn_like(*to_tensor(t))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_full_like(TensorHandle t, double val) {
    CLEAR_ERR();
    try { return from_tensor(torch::full_like(*to_tensor(t), val)); } CATCH_ERR(0)
}

// ============================================================
// 4D tensor support
// ============================================================

extern "C" TensorHandle omni_torch_zeros_4d(int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t dtype) {
    CLEAR_ERR();
    try { return from_tensor(torch::zeros({d0, d1, d2, d3}, to_dtype(dtype))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_randn_4d(int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    CLEAR_ERR();
    try { return from_tensor(torch::randn({d0, d1, d2, d3})); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_reshape_4d(TensorHandle a, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(a)->reshape({d0, d1, d2, d3})); } CATCH_ERR(0)
}

// ============================================================
// Additional math
// ============================================================

extern "C" TensorHandle omni_torch_softplus(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::nn::functional::softplus(*to_tensor(t))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_silu(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::silu(*to_tensor(t))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_gelu(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::gelu(*to_tensor(t))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_floor(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::floor(*to_tensor(t))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_ceil(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::ceil(*to_tensor(t))); } CATCH_ERR(0)
}

// ============================================================
// Batch / advanced ops
// ============================================================

extern "C" TensorHandle omni_torch_bmm(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try { return from_tensor(torch::bmm(*to_tensor(a), *to_tensor(b))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_cumsum(TensorHandle t, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(t)->cumsum(dim)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_permute_3d(TensorHandle t, int64_t d0, int64_t d1, int64_t d2) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(t)->permute({d0, d1, d2})); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_permute_4d(TensorHandle t, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(t)->permute({d0, d1, d2, d3})); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_expand_as(TensorHandle t, TensorHandle other) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(t)->expand_as(*to_tensor(other))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_repeat_4d(TensorHandle t, int64_t r0, int64_t r1, int64_t r2, int64_t r3) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(t)->repeat({r0, r1, r2, r3})); } CATCH_ERR(0)
}

// ============================================================
// Conditional / indexing
// ============================================================

extern "C" TensorHandle omni_torch_where(TensorHandle cond, TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try { return from_tensor(torch::where(*to_tensor(cond), *to_tensor(a), *to_tensor(b))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_argmax(TensorHandle t, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(t)->argmax(dim)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_argmin(TensorHandle t, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(t)->argmin(dim)); } CATCH_ERR(0)
}

// ============================================================
// Embedding
// ============================================================

extern "C" TensorHandle omni_torch_embedding(TensorHandle weight, TensorHandle indices) {
    CLEAR_ERR();
    try { return from_tensor(torch::embedding(*to_tensor(weight), *to_tensor(indices))); } CATCH_ERR(0)
}

// ============================================================
// Variance / std
// ============================================================

extern "C" TensorHandle omni_torch_var_dim(TensorHandle t, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(t)->var(dim, /*unbiased=*/true, /*keepdim=*/false)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_std_dim(TensorHandle t, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(to_tensor(t)->std(dim, /*unbiased=*/true, /*keepdim=*/false)); } CATCH_ERR(0)
}

// ============================================================
// In-place operations
// ============================================================

extern "C" void omni_torch_add_inplace(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try { to_tensor(a)->add_(*to_tensor(b)); } CATCH_ERR()
}

extern "C" void omni_torch_mul_inplace(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try { to_tensor(a)->mul_(*to_tensor(b)); } CATCH_ERR()
}

extern "C" void omni_torch_fill_f64(TensorHandle t, double val) {
    CLEAR_ERR();
    try { to_tensor(t)->fill_(val); } CATCH_ERR()
}

extern "C" void omni_torch_clamp_inplace(TensorHandle t, double min_val, double max_val) {
    CLEAR_ERR();
    try { to_tensor(t)->clamp_(min_val, max_val); } CATCH_ERR()
}

extern "C" void omni_torch_fill_missing_inplace(TensorHandle t, double nan, double posinf, double neginf) {
    CLEAR_ERR();
    try { to_tensor(t)->copy_(torch::nan_to_num(*to_tensor(t), nan, posinf, neginf)); } CATCH_ERR()
}

extern "C" void omni_torch_set_num_threads(int64_t threads) {
    CLEAR_ERR();
    try {
        if (threads <= 0) {
            throw std::runtime_error("thread count must be > 0");
        }
        at::set_num_threads(threads);
    } CATCH_ERR()
}

// ============================================================
// Complex ops (for FFT domain)
// ============================================================

extern "C" TensorHandle omni_torch_view_as_real(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::view_as_real(*to_tensor(t))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_view_as_complex(TensorHandle t) {
    CLEAR_ERR();
    try { return from_tensor(torch::view_as_complex(*to_tensor(t))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_complex_mul(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try {
        auto& ta = *to_tensor(a);
        auto& tb = *to_tensor(b);
        // If both complex, element-wise multiply
        if (ta.is_complex() && tb.is_complex()) {
            return from_tensor(ta * tb);
        }
        // Real * complex or complex * real: broadcast multiply
        return from_tensor(ta * tb);
    } CATCH_ERR(0)
}

// ============================================================
// Normalization
// ============================================================

extern "C" TensorHandle omni_torch_layer_norm(TensorHandle t, int64_t norm_dim) {
    CLEAR_ERR();
    try {
        auto& x = *to_tensor(t);
        auto shape = x.sizes();
        std::vector<int64_t> norm_shape(shape.begin() + norm_dim, shape.end());
        return from_tensor(torch::layer_norm(x, norm_shape));
    } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_rms_norm(TensorHandle t, int64_t norm_dim) {
    CLEAR_ERR();
    try {
        auto& x = *to_tensor(t);
        auto rms = x.pow(2).mean(norm_dim, true).sqrt().clamp_min(1e-8);
        return from_tensor(x / rms);
    } CATCH_ERR(0)
}

// ============================================================
// Conv1d
// ============================================================

extern "C" TensorHandle omni_torch_conv1d(TensorHandle input, TensorHandle weight, TensorHandle bias, int64_t stride, int64_t padding) {
    CLEAR_ERR();
    try {
        auto b = (bias != 0) ? c10::optional<at::Tensor>(*to_tensor(bias)) : c10::nullopt;
        return from_tensor(torch::conv1d(*to_tensor(input), *to_tensor(weight), b, stride, padding));
    } CATCH_ERR(0)
}

// ============================================================
// Dropout
// ============================================================

extern "C" TensorHandle omni_torch_dropout(TensorHandle t, double p, int64_t training) {
    CLEAR_ERR();
    try {
        return from_tensor(torch::dropout(*to_tensor(t), p, training != 0));
    } CATCH_ERR(0)
}
