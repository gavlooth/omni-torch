#ifndef TORCH_SHIM_H
#define TORCH_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle = pointer cast to int64_t for FFI
typedef int64_t TensorHandle;
typedef int64_t ScalarHandle;

// Dtype constants (match ATen ScalarType enum)
#define OMNI_FLOAT32 6
#define OMNI_FLOAT64 7
#define OMNI_INT32   3
#define OMNI_INT64   4

// === Error Handling ===
// Returns last error message, or NULL if no error. Caller does NOT free.
const char* omni_torch_last_error(void);

// === Tensor Creation ===
TensorHandle omni_torch_zeros_1d(int64_t d0, int64_t dtype);
TensorHandle omni_torch_zeros_2d(int64_t d0, int64_t d1, int64_t dtype);
TensorHandle omni_torch_zeros_3d(int64_t d0, int64_t d1, int64_t d2, int64_t dtype);
TensorHandle omni_torch_ones_1d(int64_t d0, int64_t dtype);
TensorHandle omni_torch_ones_2d(int64_t d0, int64_t d1, int64_t dtype);
TensorHandle omni_torch_ones_3d(int64_t d0, int64_t d1, int64_t d2, int64_t dtype);
TensorHandle omni_torch_rand_1d(int64_t d0);
TensorHandle omni_torch_rand_2d(int64_t d0, int64_t d1);
TensorHandle omni_torch_full_1d(int64_t d0, int64_t dtype);  // fill value via separate call
TensorHandle omni_torch_full_2d(int64_t d0, int64_t d1, int64_t dtype);
TensorHandle omni_torch_arange(int64_t start, int64_t end, int64_t step);
TensorHandle omni_torch_linspace(int64_t n);  // 0 to 1, n steps, float32
TensorHandle omni_torch_eye(int64_t n);
TensorHandle omni_torch_from_blob_f32(int64_t data_ptr, int64_t numel);
TensorHandle omni_torch_from_blob_f64(int64_t data_ptr, int64_t numel);

// Float-arg creation (separate to avoid mixed-type FFI limitation)
TensorHandle omni_torch_full_scalar_1d(int64_t d0, int64_t dtype);  // see fill below

// === Tensor Info ===
int64_t omni_torch_dim(TensorHandle t);
int64_t omni_torch_numel(TensorHandle t);
int64_t omni_torch_shape(TensorHandle t, int64_t dim_idx);
int64_t omni_torch_dtype(TensorHandle t);
int64_t omni_torch_is_contiguous(TensorHandle t);
int64_t omni_torch_data_ptr(TensorHandle t);

// === Element Access ===
// For float tensors: index → double
double  omni_torch_item_f64(TensorHandle t);
int64_t omni_torch_item_i64(TensorHandle t);

// Indexed getters (1d, 2d)
double  omni_torch_get_1d_f64(TensorHandle t, int64_t i);
double  omni_torch_get_2d_f64(TensorHandle t, int64_t i, int64_t j);
int64_t omni_torch_get_1d_i64(TensorHandle t, int64_t i);
int64_t omni_torch_get_2d_i64(TensorHandle t, int64_t i, int64_t j);

// === Arithmetic (tensor, tensor → tensor) ===
TensorHandle omni_torch_add(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_sub(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_mul(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_div(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_neg(TensorHandle a);
TensorHandle omni_torch_abs(TensorHandle a);

// Scalar arithmetic (tensor op scalar_double → tensor)
// Uses double args — call via double FFI path
TensorHandle omni_torch_add_scalar_i(TensorHandle a, int64_t s);
TensorHandle omni_torch_sub_scalar_i(TensorHandle a, int64_t s);
TensorHandle omni_torch_mul_scalar_i(TensorHandle a, int64_t s);
TensorHandle omni_torch_div_scalar_i(TensorHandle a, int64_t s);

// === Linear Algebra ===
TensorHandle omni_torch_matmul(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_mm(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_mv(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_dot(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_transpose(TensorHandle a, int64_t d0, int64_t d1);

// === Reductions ===
TensorHandle omni_torch_sum(TensorHandle a);
TensorHandle omni_torch_mean(TensorHandle a);
TensorHandle omni_torch_min(TensorHandle a);
TensorHandle omni_torch_max(TensorHandle a);
TensorHandle omni_torch_sum_dim(TensorHandle a, int64_t dim);
TensorHandle omni_torch_mean_dim(TensorHandle a, int64_t dim);

// === Shape Operations ===
TensorHandle omni_torch_reshape_1d(TensorHandle a, int64_t d0);
TensorHandle omni_torch_reshape_2d(TensorHandle a, int64_t d0, int64_t d1);
TensorHandle omni_torch_reshape_3d(TensorHandle a, int64_t d0, int64_t d1, int64_t d2);
TensorHandle omni_torch_squeeze(TensorHandle a);
TensorHandle omni_torch_unsqueeze(TensorHandle a, int64_t dim);
TensorHandle omni_torch_flatten(TensorHandle a);
TensorHandle omni_torch_cat_2(TensorHandle a, TensorHandle b, int64_t dim);
TensorHandle omni_torch_stack_2(TensorHandle a, TensorHandle b, int64_t dim);
TensorHandle omni_torch_select(TensorHandle a, int64_t dim, int64_t index);
TensorHandle omni_torch_slice(TensorHandle a, int64_t dim, int64_t start, int64_t end);
TensorHandle omni_torch_index_select(TensorHandle a, int64_t dim, TensorHandle idx);
TensorHandle omni_torch_clone(TensorHandle a);
TensorHandle omni_torch_contiguous(TensorHandle a);
TensorHandle omni_torch_to_dtype(TensorHandle a, int64_t dtype);

// === Math ===
TensorHandle omni_torch_exp(TensorHandle a);
TensorHandle omni_torch_log(TensorHandle a);
TensorHandle omni_torch_sqrt(TensorHandle a);
TensorHandle omni_torch_pow_scalar(TensorHandle a, int64_t exp);
TensorHandle omni_torch_sin(TensorHandle a);
TensorHandle omni_torch_cos(TensorHandle a);
TensorHandle omni_torch_tanh(TensorHandle a);
TensorHandle omni_torch_sigmoid(TensorHandle a);

// === Comparison ===
TensorHandle omni_torch_eq(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_lt(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_gt(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_le(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_ge(TensorHandle a, TensorHandle b);

// === NN Layers ===
TensorHandle omni_torch_relu(TensorHandle a);
TensorHandle omni_torch_softmax(TensorHandle a, int64_t dim);
TensorHandle omni_torch_log_softmax(TensorHandle a, int64_t dim);

// === In-Place Fill ===
void omni_torch_fill_i64(TensorHandle a, int64_t val);

// === Memory ===
void omni_torch_free(TensorHandle t);

// === Display ===
void omni_torch_print(TensorHandle t);
int64_t omni_torch_to_string(TensorHandle t, int64_t buf_ptr, int64_t buf_len);

#ifdef __cplusplus
}
#endif

#endif // TORCH_SHIM_H
