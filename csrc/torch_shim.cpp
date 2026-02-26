#include "torch_shim.h"
#include <torch/torch.h>
#include <sstream>
#include <cstring>

// Thread-local error buffer
static thread_local std::string g_last_error;
static thread_local bool g_has_error = false;

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

static inline TensorHandle from_tensor(at::Tensor t) {
    auto* p = new at::Tensor(std::move(t));
    return reinterpret_cast<TensorHandle>(p);
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

// === Element Access ===

extern "C" double omni_torch_item_f64(TensorHandle t) {
    CLEAR_ERR();
    try { return to_tensor(t)->item<double>(); } CATCH_ERR(0.0)
}

extern "C" int64_t omni_torch_item_i64(TensorHandle t) {
    CLEAR_ERR();
    try { return to_tensor(t)->item<int64_t>(); } CATCH_ERR(0)
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
    try { return to_tensor(t)->index({i}).item<int64_t>(); } CATCH_ERR(0)
}

extern "C" int64_t omni_torch_get_2d_i64(TensorHandle t, int64_t i, int64_t j) {
    CLEAR_ERR();
    try { return to_tensor(t)->index({i, j}).item<int64_t>(); } CATCH_ERR(0)
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

// === Linear Algebra ===

extern "C" TensorHandle omni_torch_matmul(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try { return from_tensor(torch::matmul(*to_tensor(a), *to_tensor(b))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_mm(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try { return from_tensor(torch::mm(*to_tensor(a), *to_tensor(b))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_mv(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try { return from_tensor(torch::mv(*to_tensor(a), *to_tensor(b))); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_dot(TensorHandle a, TensorHandle b) {
    CLEAR_ERR();
    try { return from_tensor(torch::dot(*to_tensor(a), *to_tensor(b))); } CATCH_ERR(0)
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
    try { return from_tensor(to_tensor(a)->mean()); } CATCH_ERR(0)
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
    try { return from_tensor(to_tensor(a)->mean(dim)); } CATCH_ERR(0)
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
    try { return from_tensor(torch::fn(*to_tensor(a))); } CATCH_ERR(0) \
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
    try { return from_tensor(torch::softmax(*to_tensor(a), dim)); } CATCH_ERR(0)
}

extern "C" TensorHandle omni_torch_log_softmax(TensorHandle a, int64_t dim) {
    CLEAR_ERR();
    try { return from_tensor(torch::log_softmax(*to_tensor(a), dim)); } CATCH_ERR(0)
}

// === In-Place ===

extern "C" void omni_torch_fill_i64(TensorHandle a, int64_t val) {
    CLEAR_ERR();
    try { to_tensor(a)->fill_(val); } catch (...) {}
}

// === Memory ===

extern "C" void omni_torch_free(TensorHandle t) {
    if (t) delete to_tensor(t);
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
