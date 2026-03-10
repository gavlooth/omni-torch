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

// === Memory and runtime controls ===
int64_t omni_torch_get_num_threads(void);
void omni_torch_set_memory_limit_bytes(int64_t bytes);
int64_t omni_torch_get_memory_limit_bytes(void);
int64_t omni_torch_get_memory_allocated_bytes(void);
int64_t omni_torch_get_memory_peak_bytes(void);

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
TensorHandle omni_torch_is_not_a_number(TensorHandle t);
TensorHandle omni_torch_is_infinite(TensorHandle t);
TensorHandle omni_torch_is_finite(TensorHandle t);
TensorHandle omni_torch_nan_to_num(TensorHandle t, double nan, double posinf, double neginf);
TensorHandle omni_torch_median(TensorHandle t);
TensorHandle omni_torch_median_dim(TensorHandle t, int64_t dim);
TensorHandle omni_torch_quantile(TensorHandle t, double q);
TensorHandle omni_torch_quantile_dim(TensorHandle t, double q, int64_t dim);
TensorHandle omni_torch_covariance(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_correlation(TensorHandle a, TensorHandle b);
TensorHandle omni_torch_histogram(TensorHandle t, int64_t bins, double min, double max);
TensorHandle omni_torch_unique_values_with_counts(TensorHandle t);
TensorHandle omni_torch_count_distinct(TensorHandle t);

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
TensorHandle omni_torch_min_dim(TensorHandle a, int64_t dim);
TensorHandle omni_torch_max_dim(TensorHandle a, int64_t dim);

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

// Scalar arithmetic: tensor op double → tensor (requires libffi for mixed types)
TensorHandle omni_torch_add_scalar_f(TensorHandle a, double s);
TensorHandle omni_torch_sub_scalar_f(TensorHandle a, double s);
TensorHandle omni_torch_mul_scalar_f(TensorHandle a, double s);
TensorHandle omni_torch_div_scalar_f(TensorHandle a, double s);

// === Gaussian Random ===
TensorHandle omni_torch_randn_1d(int64_t d0);
TensorHandle omni_torch_randn_2d(int64_t d0, int64_t d1);

// === Math ===
TensorHandle omni_torch_exp(TensorHandle a);
TensorHandle omni_torch_log(TensorHandle a);
TensorHandle omni_torch_sqrt(TensorHandle a);
TensorHandle omni_torch_pow_scalar(TensorHandle a, int64_t exp);
TensorHandle omni_torch_pow_scalar_f(TensorHandle a, double exp);
TensorHandle omni_torch_sin(TensorHandle a);
TensorHandle omni_torch_cos(TensorHandle a);
TensorHandle omni_torch_tanh(TensorHandle a);
TensorHandle omni_torch_sigmoid(TensorHandle a);

// === Clamp ===
TensorHandle omni_torch_clamp(TensorHandle a, double min_val, double max_val);

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

// === 3D random ===
TensorHandle omni_torch_rand_3d(int64_t d0, int64_t d1, int64_t d2);
TensorHandle omni_torch_randn_3d(int64_t d0, int64_t d1, int64_t d2);

// === FFT (for Wave-PDE spectral Laplacian) ===
TensorHandle omni_torch_fft_rfft(TensorHandle t, int64_t dim);
TensorHandle omni_torch_fft_irfft(TensorHandle t, int64_t n, int64_t dim);
TensorHandle omni_torch_fft_fftfreq(int64_t n);   // returns frequency vector

// === Like-constructors ===
TensorHandle omni_torch_zeros_like(TensorHandle t);
TensorHandle omni_torch_ones_like(TensorHandle t);
TensorHandle omni_torch_randn_like(TensorHandle t);
TensorHandle omni_torch_full_like(TensorHandle t, double val);

// === 4D tensor support ===
TensorHandle omni_torch_zeros_4d(int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t dtype);
TensorHandle omni_torch_randn_4d(int64_t d0, int64_t d1, int64_t d2, int64_t d3);
TensorHandle omni_torch_reshape_4d(TensorHandle a, int64_t d0, int64_t d1, int64_t d2, int64_t d3);

// === Additional math ===
TensorHandle omni_torch_softplus(TensorHandle t);   // log(1 + exp(x))
TensorHandle omni_torch_silu(TensorHandle t);       // x * sigmoid(x)
TensorHandle omni_torch_gelu(TensorHandle t);
TensorHandle omni_torch_floor(TensorHandle t);
TensorHandle omni_torch_ceil(TensorHandle t);

// === Batch/advanced ops ===
TensorHandle omni_torch_bmm(TensorHandle a, TensorHandle b);           // batch matmul
TensorHandle omni_torch_cumsum(TensorHandle t, int64_t dim);
TensorHandle omni_torch_permute_3d(TensorHandle t, int64_t d0, int64_t d1, int64_t d2);
TensorHandle omni_torch_permute_4d(TensorHandle t, int64_t d0, int64_t d1, int64_t d2, int64_t d3);
TensorHandle omni_torch_expand_as(TensorHandle t, TensorHandle other);
TensorHandle omni_torch_repeat_4d(TensorHandle t, int64_t r0, int64_t r1, int64_t r2, int64_t r3);

// === Conditional / indexing ===
TensorHandle omni_torch_where(TensorHandle cond, TensorHandle a, TensorHandle b);
TensorHandle omni_torch_argmax(TensorHandle t, int64_t dim);
TensorHandle omni_torch_argmin(TensorHandle t, int64_t dim);

// === Embedding ===
TensorHandle omni_torch_embedding(TensorHandle weight, TensorHandle indices);

// === Variance / std ===
TensorHandle omni_torch_var_dim(TensorHandle t, int64_t dim);
TensorHandle omni_torch_std_dim(TensorHandle t, int64_t dim);

// === In-place / mutation ===
void omni_torch_add_inplace(TensorHandle a, TensorHandle b);
void omni_torch_mul_inplace(TensorHandle a, TensorHandle b);
void omni_torch_fill_f64(TensorHandle t, double val);
void omni_torch_clamp_inplace(TensorHandle t, double min_val, double max_val);
void omni_torch_fill_missing_inplace(TensorHandle t, double nan, double posinf, double neginf);
void omni_torch_set_num_threads(int64_t threads);

// === Complex number ops (for FFT) ===
TensorHandle omni_torch_complex_mul(TensorHandle a, TensorHandle b);   // element-wise complex multiply
TensorHandle omni_torch_view_as_real(TensorHandle t);
TensorHandle omni_torch_view_as_complex(TensorHandle t);

// === Norm ===
TensorHandle omni_torch_layer_norm(TensorHandle t, int64_t norm_dim);  // simplified: last dim
TensorHandle omni_torch_rms_norm(TensorHandle t, int64_t norm_dim);    // root mean square norm

// === Conv1d (for 1x1 convolutions in Wave-PDE) ===
TensorHandle omni_torch_conv1d(TensorHandle input, TensorHandle weight, TensorHandle bias, int64_t stride, int64_t padding);

// === Dropout (inference only: identity; train: random mask) ===
TensorHandle omni_torch_dropout(TensorHandle t, double p, int64_t training);

#ifdef __cplusplus
}
#endif

#endif // TORCH_SHIM_H
