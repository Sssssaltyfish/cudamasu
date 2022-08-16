#pragma once
#ifndef CUDAMASU_H_
#define CUDAMASU_H_

#ifndef __CUDACC__

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

//* define compile options
#define __device__
#define __global__
#define __host__
#define __shared__

#define __DEVICE__                                                             \
  inline __attribute__((always_inline)) __attribute__((weak)) __device__

//* define predicate functions
__DEVICE__ unsigned int __isGlobal(const void *p);
__DEVICE__ unsigned int __isShared(const void *p);
__DEVICE__ unsigned int __isConstant(const void *p);
__DEVICE__ unsigned int __isLocal(const void *p);

//* define cuda-side system calls
extern "C" {
// Device-side CUDA system calls.
// http://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#system-calls
__device__ int vprintf(const char *, const char *);

// __assertfail() used to have a `noreturn` attribute. Unfortunately that
// contributed to triggering the longstanding bug in ptxas when assert was used
// in sufficiently convoluted code. See
// https://bugs.llvm.org/show_bug.cgi?id=27738 for the details.
__device__ void __assertfail(const char *__message, const char *__file,
                             unsigned __line, const char *__function,
                             size_t __charSize);

// In order for standard assert() macro on linux to work we need to
// provide device-side __assert_fail()
__device__ static inline void __assert_fail(const char *__message,
                                            const char *__file, unsigned __line,
                                            const char *__function);

// Clang will convert printf into vprintf, but we still need
// device-side declaration for it.
__device__ int printf(const char *, ...);
} // extern "C"

//* define cuda libdevice declares
__DEVICE__ int __nv_abs(int __a);
__DEVICE__ double __nv_acos(double __a);
__DEVICE__ float __nv_acosf(float __a);
__DEVICE__ double __nv_acosh(double __a);
__DEVICE__ float __nv_acoshf(float __a);
__DEVICE__ double __nv_asin(double __a);
__DEVICE__ float __nv_asinf(float __a);
__DEVICE__ double __nv_asinh(double __a);
__DEVICE__ float __nv_asinhf(float __a);
__DEVICE__ double __nv_atan2(double __a, double __b);
__DEVICE__ float __nv_atan2f(float __a, float __b);
__DEVICE__ double __nv_atan(double __a);
__DEVICE__ float __nv_atanf(float __a);
__DEVICE__ double __nv_atanh(double __a);
__DEVICE__ float __nv_atanhf(float __a);
__DEVICE__ int __nv_brev(int __a);
__DEVICE__ long long __nv_brevll(long long __a);
__DEVICE__ int __nv_byte_perm(int __a, int __b, int __c);
__DEVICE__ double __nv_cbrt(double __a);
__DEVICE__ float __nv_cbrtf(float __a);
__DEVICE__ double __nv_ceil(double __a);
__DEVICE__ float __nv_ceilf(float __a);
__DEVICE__ int __nv_clz(int __a);
__DEVICE__ int __nv_clzll(long long __a);
__DEVICE__ double __nv_copysign(double __a, double __b);
__DEVICE__ float __nv_copysignf(float __a, float __b);
__DEVICE__ double __nv_cos(double __a);
__DEVICE__ float __nv_cosf(float __a);
__DEVICE__ double __nv_cosh(double __a);
__DEVICE__ float __nv_coshf(float __a);
__DEVICE__ double __nv_cospi(double __a);
__DEVICE__ float __nv_cospif(float __a);
__DEVICE__ double __nv_cyl_bessel_i0(double __a);
__DEVICE__ float __nv_cyl_bessel_i0f(float __a);
__DEVICE__ double __nv_cyl_bessel_i1(double __a);
__DEVICE__ float __nv_cyl_bessel_i1f(float __a);
__DEVICE__ double __nv_dadd_rd(double __a, double __b);
__DEVICE__ double __nv_dadd_rn(double __a, double __b);
__DEVICE__ double __nv_dadd_ru(double __a, double __b);
__DEVICE__ double __nv_dadd_rz(double __a, double __b);
__DEVICE__ double __nv_ddiv_rd(double __a, double __b);
__DEVICE__ double __nv_ddiv_rn(double __a, double __b);
__DEVICE__ double __nv_ddiv_ru(double __a, double __b);
__DEVICE__ double __nv_ddiv_rz(double __a, double __b);
__DEVICE__ double __nv_dmul_rd(double __a, double __b);
__DEVICE__ double __nv_dmul_rn(double __a, double __b);
__DEVICE__ double __nv_dmul_ru(double __a, double __b);
__DEVICE__ double __nv_dmul_rz(double __a, double __b);
__DEVICE__ float __nv_double2float_rd(double __a);
__DEVICE__ float __nv_double2float_rn(double __a);
__DEVICE__ float __nv_double2float_ru(double __a);
__DEVICE__ float __nv_double2float_rz(double __a);
__DEVICE__ int __nv_double2hiint(double __a);
__DEVICE__ int __nv_double2int_rd(double __a);
__DEVICE__ int __nv_double2int_rn(double __a);
__DEVICE__ int __nv_double2int_ru(double __a);
__DEVICE__ int __nv_double2int_rz(double __a);
__DEVICE__ long long __nv_double2ll_rd(double __a);
__DEVICE__ long long __nv_double2ll_rn(double __a);
__DEVICE__ long long __nv_double2ll_ru(double __a);
__DEVICE__ long long __nv_double2ll_rz(double __a);
__DEVICE__ int __nv_double2loint(double __a);
__DEVICE__ unsigned int __nv_double2uint_rd(double __a);
__DEVICE__ unsigned int __nv_double2uint_rn(double __a);
__DEVICE__ unsigned int __nv_double2uint_ru(double __a);
__DEVICE__ unsigned int __nv_double2uint_rz(double __a);
__DEVICE__ unsigned long long __nv_double2ull_rd(double __a);
__DEVICE__ unsigned long long __nv_double2ull_rn(double __a);
__DEVICE__ unsigned long long __nv_double2ull_ru(double __a);
__DEVICE__ unsigned long long __nv_double2ull_rz(double __a);
__DEVICE__ unsigned long long __nv_double_as_longlong(double __a);
__DEVICE__ double __nv_drcp_rd(double __a);
__DEVICE__ double __nv_drcp_rn(double __a);
__DEVICE__ double __nv_drcp_ru(double __a);
__DEVICE__ double __nv_drcp_rz(double __a);
__DEVICE__ double __nv_dsqrt_rd(double __a);
__DEVICE__ double __nv_dsqrt_rn(double __a);
__DEVICE__ double __nv_dsqrt_ru(double __a);
__DEVICE__ double __nv_dsqrt_rz(double __a);
__DEVICE__ double __nv_dsub_rd(double __a, double __b);
__DEVICE__ double __nv_dsub_rn(double __a, double __b);
__DEVICE__ double __nv_dsub_ru(double __a, double __b);
__DEVICE__ double __nv_dsub_rz(double __a, double __b);
__DEVICE__ double __nv_erfc(double __a);
__DEVICE__ float __nv_erfcf(float __a);
__DEVICE__ double __nv_erfcinv(double __a);
__DEVICE__ float __nv_erfcinvf(float __a);
__DEVICE__ double __nv_erfcx(double __a);
__DEVICE__ float __nv_erfcxf(float __a);
__DEVICE__ double __nv_erf(double __a);
__DEVICE__ float __nv_erff(float __a);
__DEVICE__ double __nv_erfinv(double __a);
__DEVICE__ float __nv_erfinvf(float __a);
__DEVICE__ double __nv_exp10(double __a);
__DEVICE__ float __nv_exp10f(float __a);
__DEVICE__ double __nv_exp2(double __a);
__DEVICE__ float __nv_exp2f(float __a);
__DEVICE__ double __nv_exp(double __a);
__DEVICE__ float __nv_expf(float __a);
__DEVICE__ double __nv_expm1(double __a);
__DEVICE__ float __nv_expm1f(float __a);
__DEVICE__ double __nv_fabs(double __a);
__DEVICE__ float __nv_fabsf(float __a);
__DEVICE__ float __nv_fadd_rd(float __a, float __b);
__DEVICE__ float __nv_fadd_rn(float __a, float __b);
__DEVICE__ float __nv_fadd_ru(float __a, float __b);
__DEVICE__ float __nv_fadd_rz(float __a, float __b);
__DEVICE__ float __nv_fast_cosf(float __a);
__DEVICE__ float __nv_fast_exp10f(float __a);
__DEVICE__ float __nv_fast_expf(float __a);
__DEVICE__ float __nv_fast_fdividef(float __a, float __b);
__DEVICE__ float __nv_fast_log10f(float __a);
__DEVICE__ float __nv_fast_log2f(float __a);
__DEVICE__ float __nv_fast_logf(float __a);
__DEVICE__ float __nv_fast_powf(float __a, float __b);
__DEVICE__ void __nv_fast_sincosf(float __a, float *__s, float *__c);
__DEVICE__ float __nv_fast_sinf(float __a);
__DEVICE__ float __nv_fast_tanf(float __a);
__DEVICE__ double __nv_fdim(double __a, double __b);
__DEVICE__ float __nv_fdimf(float __a, float __b);
__DEVICE__ float __nv_fdiv_rd(float __a, float __b);
__DEVICE__ float __nv_fdiv_rn(float __a, float __b);
__DEVICE__ float __nv_fdiv_ru(float __a, float __b);
__DEVICE__ float __nv_fdiv_rz(float __a, float __b);
__DEVICE__ int __nv_ffs(int __a);
__DEVICE__ int __nv_ffsll(long long __a);
__DEVICE__ int __nv_finitef(float __a);
__DEVICE__ unsigned short __nv_float2half_rn(float __a);
__DEVICE__ int __nv_float2int_rd(float __a);
__DEVICE__ int __nv_float2int_rn(float __a);
__DEVICE__ int __nv_float2int_ru(float __a);
__DEVICE__ int __nv_float2int_rz(float __a);
__DEVICE__ long long __nv_float2ll_rd(float __a);
__DEVICE__ long long __nv_float2ll_rn(float __a);
__DEVICE__ long long __nv_float2ll_ru(float __a);
__DEVICE__ long long __nv_float2ll_rz(float __a);
__DEVICE__ unsigned int __nv_float2uint_rd(float __a);
__DEVICE__ unsigned int __nv_float2uint_rn(float __a);
__DEVICE__ unsigned int __nv_float2uint_ru(float __a);
__DEVICE__ unsigned int __nv_float2uint_rz(float __a);
__DEVICE__ unsigned long long __nv_float2ull_rd(float __a);
__DEVICE__ unsigned long long __nv_float2ull_rn(float __a);
__DEVICE__ unsigned long long __nv_float2ull_ru(float __a);
__DEVICE__ unsigned long long __nv_float2ull_rz(float __a);
__DEVICE__ int __nv_float_as_int(float __a);
__DEVICE__ unsigned int __nv_float_as_uint(float __a);
__DEVICE__ double __nv_floor(double __a);
__DEVICE__ float __nv_floorf(float __a);
__DEVICE__ double __nv_fma(double __a, double __b, double __c);
__DEVICE__ float __nv_fmaf(float __a, float __b, float __c);
__DEVICE__ float __nv_fmaf_ieee_rd(float __a, float __b, float __c);
__DEVICE__ float __nv_fmaf_ieee_rn(float __a, float __b, float __c);
__DEVICE__ float __nv_fmaf_ieee_ru(float __a, float __b, float __c);
__DEVICE__ float __nv_fmaf_ieee_rz(float __a, float __b, float __c);
__DEVICE__ float __nv_fmaf_rd(float __a, float __b, float __c);
__DEVICE__ float __nv_fmaf_rn(float __a, float __b, float __c);
__DEVICE__ float __nv_fmaf_ru(float __a, float __b, float __c);
__DEVICE__ float __nv_fmaf_rz(float __a, float __b, float __c);
__DEVICE__ double __nv_fma_rd(double __a, double __b, double __c);
__DEVICE__ double __nv_fma_rn(double __a, double __b, double __c);
__DEVICE__ double __nv_fma_ru(double __a, double __b, double __c);
__DEVICE__ double __nv_fma_rz(double __a, double __b, double __c);
__DEVICE__ double __nv_fmax(double __a, double __b);
__DEVICE__ float __nv_fmaxf(float __a, float __b);
__DEVICE__ double __nv_fmin(double __a, double __b);
__DEVICE__ float __nv_fminf(float __a, float __b);
__DEVICE__ double __nv_fmod(double __a, double __b);
__DEVICE__ float __nv_fmodf(float __a, float __b);
__DEVICE__ float __nv_fmul_rd(float __a, float __b);
__DEVICE__ float __nv_fmul_rn(float __a, float __b);
__DEVICE__ float __nv_fmul_ru(float __a, float __b);
__DEVICE__ float __nv_fmul_rz(float __a, float __b);
__DEVICE__ float __nv_frcp_rd(float __a);
__DEVICE__ float __nv_frcp_rn(float __a);
__DEVICE__ float __nv_frcp_ru(float __a);
__DEVICE__ float __nv_frcp_rz(float __a);
__DEVICE__ double __nv_frexp(double __a, int *__b);
__DEVICE__ float __nv_frexpf(float __a, int *__b);
__DEVICE__ float __nv_frsqrt_rn(float __a);
__DEVICE__ float __nv_fsqrt_rd(float __a);
__DEVICE__ float __nv_fsqrt_rn(float __a);
__DEVICE__ float __nv_fsqrt_ru(float __a);
__DEVICE__ float __nv_fsqrt_rz(float __a);
__DEVICE__ float __nv_fsub_rd(float __a, float __b);
__DEVICE__ float __nv_fsub_rn(float __a, float __b);
__DEVICE__ float __nv_fsub_ru(float __a, float __b);
__DEVICE__ float __nv_fsub_rz(float __a, float __b);
__DEVICE__ int __nv_hadd(int __a, int __b);
__DEVICE__ float __nv_half2float(unsigned short __h);
__DEVICE__ double __nv_hiloint2double(int __a, int __b);
__DEVICE__ double __nv_hypot(double __a, double __b);
__DEVICE__ float __nv_hypotf(float __a, float __b);
__DEVICE__ int __nv_ilogb(double __a);
__DEVICE__ int __nv_ilogbf(float __a);
__DEVICE__ double __nv_int2double_rn(int __a);
__DEVICE__ float __nv_int2float_rd(int __a);
__DEVICE__ float __nv_int2float_rn(int __a);
__DEVICE__ float __nv_int2float_ru(int __a);
__DEVICE__ float __nv_int2float_rz(int __a);
__DEVICE__ float __nv_int_as_float(int __a);
__DEVICE__ int __nv_isfinited(double __a);
__DEVICE__ int __nv_isinfd(double __a);
__DEVICE__ int __nv_isinff(float __a);
__DEVICE__ int __nv_isnand(double __a);
__DEVICE__ int __nv_isnanf(float __a);
__DEVICE__ double __nv_j0(double __a);
__DEVICE__ float __nv_j0f(float __a);
__DEVICE__ double __nv_j1(double __a);
__DEVICE__ float __nv_j1f(float __a);
__DEVICE__ float __nv_jnf(int __a, float __b);
__DEVICE__ double __nv_jn(int __a, double __b);
__DEVICE__ double __nv_ldexp(double __a, int __b);
__DEVICE__ float __nv_ldexpf(float __a, int __b);
__DEVICE__ double __nv_lgamma(double __a);
__DEVICE__ float __nv_lgammaf(float __a);
__DEVICE__ double __nv_ll2double_rd(long long __a);
__DEVICE__ double __nv_ll2double_rn(long long __a);
__DEVICE__ double __nv_ll2double_ru(long long __a);
__DEVICE__ double __nv_ll2double_rz(long long __a);
__DEVICE__ float __nv_ll2float_rd(long long __a);
__DEVICE__ float __nv_ll2float_rn(long long __a);
__DEVICE__ float __nv_ll2float_ru(long long __a);
__DEVICE__ float __nv_ll2float_rz(long long __a);
__DEVICE__ long long __nv_llabs(long long __a);
__DEVICE__ long long __nv_llmax(long long __a, long long __b);
__DEVICE__ long long __nv_llmin(long long __a, long long __b);
__DEVICE__ long long __nv_llrint(double __a);
__DEVICE__ long long __nv_llrintf(float __a);
__DEVICE__ long long __nv_llround(double __a);
__DEVICE__ long long __nv_llroundf(float __a);
__DEVICE__ double __nv_log10(double __a);
__DEVICE__ float __nv_log10f(float __a);
__DEVICE__ double __nv_log1p(double __a);
__DEVICE__ float __nv_log1pf(float __a);
__DEVICE__ double __nv_log2(double __a);
__DEVICE__ float __nv_log2f(float __a);
__DEVICE__ double __nv_logb(double __a);
__DEVICE__ float __nv_logbf(float __a);
__DEVICE__ double __nv_log(double __a);
__DEVICE__ float __nv_logf(float __a);
__DEVICE__ double __nv_longlong_as_double(long long __a);
__DEVICE__ int __nv_max(int __a, int __b);
__DEVICE__ int __nv_min(int __a, int __b);
__DEVICE__ double __nv_modf(double __a, double *__b);
__DEVICE__ float __nv_modff(float __a, float *__b);
__DEVICE__ int __nv_mul24(int __a, int __b);
__DEVICE__ long long __nv_mul64hi(long long __a, long long __b);
__DEVICE__ int __nv_mulhi(int __a, int __b);
__DEVICE__ double __nv_nan(const signed char *__a);
__DEVICE__ float __nv_nanf(const signed char *__a);
__DEVICE__ double __nv_nearbyint(double __a);
__DEVICE__ float __nv_nearbyintf(float __a);
__DEVICE__ double __nv_nextafter(double __a, double __b);
__DEVICE__ float __nv_nextafterf(float __a, float __b);
__DEVICE__ double __nv_norm3d(double __a, double __b, double __c);
__DEVICE__ float __nv_norm3df(float __a, float __b, float __c);
__DEVICE__ double __nv_norm4d(double __a, double __b, double __c, double __d);
__DEVICE__ float __nv_norm4df(float __a, float __b, float __c, float __d);
__DEVICE__ double __nv_normcdf(double __a);
__DEVICE__ float __nv_normcdff(float __a);
__DEVICE__ double __nv_normcdfinv(double __a);
__DEVICE__ float __nv_normcdfinvf(float __a);
__DEVICE__ float __nv_normf(int __a, const float *__b);
__DEVICE__ double __nv_norm(int __a, const double *__b);
__DEVICE__ int __nv_popc(int __a);
__DEVICE__ int __nv_popcll(long long __a);
__DEVICE__ double __nv_pow(double __a, double __b);
__DEVICE__ float __nv_powf(float __a, float __b);
__DEVICE__ double __nv_powi(double __a, int __b);
__DEVICE__ float __nv_powif(float __a, int __b);
__DEVICE__ double __nv_rcbrt(double __a);
__DEVICE__ float __nv_rcbrtf(float __a);
__DEVICE__ double __nv_rcp64h(double __a);
__DEVICE__ double __nv_remainder(double __a, double __b);
__DEVICE__ float __nv_remainderf(float __a, float __b);
__DEVICE__ double __nv_remquo(double __a, double __b, int *__c);
__DEVICE__ float __nv_remquof(float __a, float __b, int *__c);
__DEVICE__ int __nv_rhadd(int __a, int __b);
__DEVICE__ double __nv_rhypot(double __a, double __b);
__DEVICE__ float __nv_rhypotf(float __a, float __b);
__DEVICE__ double __nv_rint(double __a);
__DEVICE__ float __nv_rintf(float __a);
__DEVICE__ double __nv_rnorm3d(double __a, double __b, double __c);
__DEVICE__ float __nv_rnorm3df(float __a, float __b, float __c);
__DEVICE__ double __nv_rnorm4d(double __a, double __b, double __c, double __d);
__DEVICE__ float __nv_rnorm4df(float __a, float __b, float __c, float __d);
__DEVICE__ float __nv_rnormf(int __a, const float *__b);
__DEVICE__ double __nv_rnorm(int __a, const double *__b);
__DEVICE__ double __nv_round(double __a);
__DEVICE__ float __nv_roundf(float __a);
__DEVICE__ double __nv_rsqrt(double __a);
__DEVICE__ float __nv_rsqrtf(float __a);
__DEVICE__ int __nv_sad(int __a, int __b, int __c);
__DEVICE__ float __nv_saturatef(float __a);
__DEVICE__ double __nv_scalbn(double __a, int __b);
__DEVICE__ float __nv_scalbnf(float __a, int __b);
__DEVICE__ int __nv_signbitd(double __a);
__DEVICE__ int __nv_signbitf(float __a);
__DEVICE__ void __nv_sincos(double __a, double *__b, double *__c);
__DEVICE__ void __nv_sincosf(float __a, float *__b, float *__c);
__DEVICE__ void __nv_sincospi(double __a, double *__b, double *__c);
__DEVICE__ void __nv_sincospif(float __a, float *__b, float *__c);
__DEVICE__ double __nv_sin(double __a);
__DEVICE__ float __nv_sinf(float __a);
__DEVICE__ double __nv_sinh(double __a);
__DEVICE__ float __nv_sinhf(float __a);
__DEVICE__ double __nv_sinpi(double __a);
__DEVICE__ float __nv_sinpif(float __a);
__DEVICE__ double __nv_sqrt(double __a);
__DEVICE__ float __nv_sqrtf(float __a);
__DEVICE__ double __nv_tan(double __a);
__DEVICE__ float __nv_tanf(float __a);
__DEVICE__ double __nv_tanh(double __a);
__DEVICE__ float __nv_tanhf(float __a);
__DEVICE__ double __nv_tgamma(double __a);
__DEVICE__ float __nv_tgammaf(float __a);
__DEVICE__ double __nv_trunc(double __a);
__DEVICE__ float __nv_truncf(float __a);
__DEVICE__ int __nv_uhadd(unsigned int __a, unsigned int __b);
__DEVICE__ double __nv_uint2double_rn(unsigned int __i);
__DEVICE__ float __nv_uint2float_rd(unsigned int __a);
__DEVICE__ float __nv_uint2float_rn(unsigned int __a);
__DEVICE__ float __nv_uint2float_ru(unsigned int __a);
__DEVICE__ float __nv_uint2float_rz(unsigned int __a);
__DEVICE__ float __nv_uint_as_float(unsigned int __a);
__DEVICE__ double __nv_ull2double_rd(unsigned long long __a);
__DEVICE__ double __nv_ull2double_rn(unsigned long long __a);
__DEVICE__ double __nv_ull2double_ru(unsigned long long __a);
__DEVICE__ double __nv_ull2double_rz(unsigned long long __a);
__DEVICE__ float __nv_ull2float_rd(unsigned long long __a);
__DEVICE__ float __nv_ull2float_rn(unsigned long long __a);
__DEVICE__ float __nv_ull2float_ru(unsigned long long __a);
__DEVICE__ float __nv_ull2float_rz(unsigned long long __a);
__DEVICE__ unsigned long long __nv_ullmax(unsigned long long __a,
                                          unsigned long long __b);
__DEVICE__ unsigned long long __nv_ullmin(unsigned long long __a,
                                          unsigned long long __b);
__DEVICE__ unsigned int __nv_umax(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __nv_umin(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __nv_umul24(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned long long __nv_umul64hi(unsigned long long __a,
                                            unsigned long long __b);
__DEVICE__ unsigned int __nv_umulhi(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __nv_urhadd(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __nv_usad(unsigned int __a, unsigned int __b,
                                  unsigned int __c);

__DEVICE__ int __nv_vabs2(int __a);
__DEVICE__ int __nv_vabs4(int __a);
__DEVICE__ int __nv_vabsdiffs2(int __a, int __b);
__DEVICE__ int __nv_vabsdiffs4(int __a, int __b);
__DEVICE__ int __nv_vabsdiffu2(int __a, int __b);
__DEVICE__ int __nv_vabsdiffu4(int __a, int __b);
__DEVICE__ int __nv_vabsss2(int __a);
__DEVICE__ int __nv_vabsss4(int __a);
__DEVICE__ int __nv_vadd2(int __a, int __b);
__DEVICE__ int __nv_vadd4(int __a, int __b);
__DEVICE__ int __nv_vaddss2(int __a, int __b);
__DEVICE__ int __nv_vaddss4(int __a, int __b);
__DEVICE__ int __nv_vaddus2(int __a, int __b);
__DEVICE__ int __nv_vaddus4(int __a, int __b);
__DEVICE__ int __nv_vavgs2(int __a, int __b);
__DEVICE__ int __nv_vavgs4(int __a, int __b);
__DEVICE__ int __nv_vavgu2(int __a, int __b);
__DEVICE__ int __nv_vavgu4(int __a, int __b);
__DEVICE__ int __nv_vcmpeq2(int __a, int __b);
__DEVICE__ int __nv_vcmpeq4(int __a, int __b);
__DEVICE__ int __nv_vcmpges2(int __a, int __b);
__DEVICE__ int __nv_vcmpges4(int __a, int __b);
__DEVICE__ int __nv_vcmpgeu2(int __a, int __b);
__DEVICE__ int __nv_vcmpgeu4(int __a, int __b);
__DEVICE__ int __nv_vcmpgts2(int __a, int __b);
__DEVICE__ int __nv_vcmpgts4(int __a, int __b);
__DEVICE__ int __nv_vcmpgtu2(int __a, int __b);
__DEVICE__ int __nv_vcmpgtu4(int __a, int __b);
__DEVICE__ int __nv_vcmples2(int __a, int __b);
__DEVICE__ int __nv_vcmples4(int __a, int __b);
__DEVICE__ int __nv_vcmpleu2(int __a, int __b);
__DEVICE__ int __nv_vcmpleu4(int __a, int __b);
__DEVICE__ int __nv_vcmplts2(int __a, int __b);
__DEVICE__ int __nv_vcmplts4(int __a, int __b);
__DEVICE__ int __nv_vcmpltu2(int __a, int __b);
__DEVICE__ int __nv_vcmpltu4(int __a, int __b);
__DEVICE__ int __nv_vcmpne2(int __a, int __b);
__DEVICE__ int __nv_vcmpne4(int __a, int __b);
__DEVICE__ int __nv_vhaddu2(int __a, int __b);
__DEVICE__ int __nv_vhaddu4(int __a, int __b);
__DEVICE__ int __nv_vmaxs2(int __a, int __b);
__DEVICE__ int __nv_vmaxs4(int __a, int __b);
__DEVICE__ int __nv_vmaxu2(int __a, int __b);
__DEVICE__ int __nv_vmaxu4(int __a, int __b);
__DEVICE__ int __nv_vmins2(int __a, int __b);
__DEVICE__ int __nv_vmins4(int __a, int __b);
__DEVICE__ int __nv_vminu2(int __a, int __b);
__DEVICE__ int __nv_vminu4(int __a, int __b);
__DEVICE__ int __nv_vneg2(int __a);
__DEVICE__ int __nv_vneg4(int __a);
__DEVICE__ int __nv_vnegss2(int __a);
__DEVICE__ int __nv_vnegss4(int __a);
__DEVICE__ int __nv_vsads2(int __a, int __b);
__DEVICE__ int __nv_vsads4(int __a, int __b);
__DEVICE__ int __nv_vsadu2(int __a, int __b);
__DEVICE__ int __nv_vsadu4(int __a, int __b);
__DEVICE__ int __nv_vseteq2(int __a, int __b);
__DEVICE__ int __nv_vseteq4(int __a, int __b);
__DEVICE__ int __nv_vsetges2(int __a, int __b);
__DEVICE__ int __nv_vsetges4(int __a, int __b);
__DEVICE__ int __nv_vsetgeu2(int __a, int __b);
__DEVICE__ int __nv_vsetgeu4(int __a, int __b);
__DEVICE__ int __nv_vsetgts2(int __a, int __b);
__DEVICE__ int __nv_vsetgts4(int __a, int __b);
__DEVICE__ int __nv_vsetgtu2(int __a, int __b);
__DEVICE__ int __nv_vsetgtu4(int __a, int __b);
__DEVICE__ int __nv_vsetles2(int __a, int __b);
__DEVICE__ int __nv_vsetles4(int __a, int __b);
__DEVICE__ int __nv_vsetleu2(int __a, int __b);
__DEVICE__ int __nv_vsetleu4(int __a, int __b);
__DEVICE__ int __nv_vsetlts2(int __a, int __b);
__DEVICE__ int __nv_vsetlts4(int __a, int __b);
__DEVICE__ int __nv_vsetltu2(int __a, int __b);
__DEVICE__ int __nv_vsetltu4(int __a, int __b);
__DEVICE__ int __nv_vsetne2(int __a, int __b);
__DEVICE__ int __nv_vsetne4(int __a, int __b);
__DEVICE__ int __nv_vsub2(int __a, int __b);
__DEVICE__ int __nv_vsub4(int __a, int __b);
__DEVICE__ int __nv_vsubss2(int __a, int __b);
__DEVICE__ int __nv_vsubss4(int __a, int __b);
__DEVICE__ int __nv_vsubus2(int __a, int __b);
__DEVICE__ int __nv_vsubus4(int __a, int __b);

__DEVICE__ double __nv_y0(double __a);
__DEVICE__ float __nv_y0f(float __a);
__DEVICE__ double __nv_y1(double __a);
__DEVICE__ float __nv_y1f(float __a);
__DEVICE__ float __nv_ynf(int __a, float __b);
__DEVICE__ double __nv_yn(int __a, double __b);

//* define vars
struct uint3 {
  unsigned int x, y, z;
};
struct dim3 {
  unsigned int x, y, z;
};

#define __DELETE = delete
#define __CUDA_DEVICE_BUILTIN(FIELD, INTRINSIC) unsigned int FIELD;

#define __CUDA_DISALLOW_BUILTINVAR_ACCESS(TypeName)                            \
  TypeName() __DELETE;                                                         \
  TypeName(const TypeName &) __DELETE;                                         \
  void operator=(const TypeName &) const __DELETE;                             \
  TypeName *operator&() const __DELETE

struct __cuda_builtin_threadIdx_t {
  __CUDA_DEVICE_BUILTIN(x, __nvvm_read_ptx_sreg_tid_x());
  __CUDA_DEVICE_BUILTIN(y, __nvvm_read_ptx_sreg_tid_y());
  __CUDA_DEVICE_BUILTIN(z, __nvvm_read_ptx_sreg_tid_z());
  // threadIdx should be convertible to uint3 (in fact in nvcc, it *is* a
  // uint3).  This function is defined after we pull in vector_types.h.
  operator dim3() const;
  operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_threadIdx_t);
};

struct __cuda_builtin_blockIdx_t {
  __CUDA_DEVICE_BUILTIN(x, __nvvm_read_ptx_sreg_ctaid_x());
  __CUDA_DEVICE_BUILTIN(y, __nvvm_read_ptx_sreg_ctaid_y());
  __CUDA_DEVICE_BUILTIN(z, __nvvm_read_ptx_sreg_ctaid_z());
  // blockIdx should be convertible to uint3 (in fact in nvcc, it *is* a
  // uint3).  This function is defined after we pull in vector_types.h.
  operator dim3() const;
  operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockIdx_t);
};

struct __cuda_builtin_blockDim_t {
  __CUDA_DEVICE_BUILTIN(x, __nvvm_read_ptx_sreg_ntid_x());
  __CUDA_DEVICE_BUILTIN(y, __nvvm_read_ptx_sreg_ntid_y());
  __CUDA_DEVICE_BUILTIN(z, __nvvm_read_ptx_sreg_ntid_z());
  // blockDim should be convertible to dim3 (in fact in nvcc, it *is* a
  // dim3).  This function is defined after we pull in vector_types.h.
  operator dim3() const;
  operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockDim_t);
};

struct __cuda_builtin_gridDim_t {
  __CUDA_DEVICE_BUILTIN(x, __nvvm_read_ptx_sreg_nctaid_x());
  __CUDA_DEVICE_BUILTIN(y, __nvvm_read_ptx_sreg_nctaid_y());
  __CUDA_DEVICE_BUILTIN(z, __nvvm_read_ptx_sreg_nctaid_z());
  // gridDim should be convertible to dim3 (in fact in nvcc, it *is* a
  // dim3).  This function is defined after we pull in vector_types.h.
  operator dim3() const;
  operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_gridDim_t);
};

#define __CUDA_BUILTIN_VAR extern const __device__ __attribute__((weak))
__CUDA_BUILTIN_VAR __cuda_builtin_threadIdx_t threadIdx;
__CUDA_BUILTIN_VAR __cuda_builtin_blockIdx_t blockIdx;
__CUDA_BUILTIN_VAR __cuda_builtin_blockDim_t blockDim;
__CUDA_BUILTIN_VAR __cuda_builtin_gridDim_t gridDim;

// warpSize should translate to read of %WARP_SZ but there's currently no
// builtin to do so. According to PTX v4.2 docs 'to date, all target
// architectures have a WARP_SZ value of 32'.
__device__ const int warpSize = 32;

//* define intrinsics
#define __SHUFFLE_TYPE(__FnName, __Type, __RetType)                            \
  inline __device__ __RetType __FnName(__RetType __val, __Type __offset,       \
                                       int __width = warpSize);
#define __MAKE_SHUFFLES(__FnName, __Type)                                      \
  __SHUFFLE_TYPE(__FnName, __Type, float)                                      \
  __SHUFFLE_TYPE(__FnName, __Type, double)                                     \
  __SHUFFLE_TYPE(__FnName, __Type, int)                                        \
  __SHUFFLE_TYPE(__FnName, __Type, unsigned int)                               \
  __SHUFFLE_TYPE(__FnName, __Type, long)                                       \
  __SHUFFLE_TYPE(__FnName, __Type, unsigned long)                              \
  __SHUFFLE_TYPE(__FnName, __Type, long long)                                  \
  __SHUFFLE_TYPE(__FnName, __Type, unsigned long long)

__MAKE_SHUFFLES(__shfl, int);
__MAKE_SHUFFLES(__shfl_up, unsigned int);
__MAKE_SHUFFLES(__shfl_down, unsigned int);
__MAKE_SHUFFLES(__shfl_xor, int);

#undef __SHUFFLE_TYPE
#undef __MAKE_SHUFFLES

#define __SHUFFLE_TYPE(__FnName, __Type, __RetType)                            \
  inline __device__ __RetType __FnName(unsigned int __mask, __RetType __val,   \
                                       __Type __offset,                        \
                                       int __width = warpSize);

#define __MAKE_SYNC_SHUFFLES(__FnName, __Type)                                 \
  __SHUFFLE_TYPE(__FnName, __Type, float)                                      \
  __SHUFFLE_TYPE(__FnName, __Type, double)                                     \
  __SHUFFLE_TYPE(__FnName, __Type, int)                                        \
  __SHUFFLE_TYPE(__FnName, __Type, unsigned int)                               \
  __SHUFFLE_TYPE(__FnName, __Type, long)                                       \
  __SHUFFLE_TYPE(__FnName, __Type, unsigned long)                              \
  __SHUFFLE_TYPE(__FnName, __Type, long long)                                  \
  __SHUFFLE_TYPE(__FnName, __Type, unsigned long long)

__MAKE_SYNC_SHUFFLES(__shfl_sync, int);
__MAKE_SYNC_SHUFFLES(__shfl_up_sync, unsigned int);
__MAKE_SYNC_SHUFFLES(__shfl_down_sync, unsigned int);
__MAKE_SYNC_SHUFFLES(__shfl_xor_sync, int);

#undef __SHUFFLE_TYPE
#undef __MAKE_SYNC_SHUFFLES

inline __device__ void __syncwarp(unsigned int mask = 0xffffffff);
inline __device__ void __barrier_sync(unsigned int id);
inline __device__ void __barrier_sync_count(unsigned int id,
                                            unsigned int count);
inline __device__ int __all_sync(unsigned int mask, int pred);
inline __device__ int __any_sync(unsigned int mask, int pred);
inline __device__ int __uni_sync(unsigned int mask, int pred);
inline __device__ unsigned int __ballot_sync(unsigned int mask, int pred);
inline __device__ unsigned int __activemask();
inline __device__ unsigned int __fns(unsigned mask, unsigned base, int offset);

inline __device__ unsigned int __match32_any_sync(unsigned int mask,
                                                  unsigned int value);

inline __device__ unsigned long long
__match64_any_sync(unsigned int mask, unsigned long long value);
inline __device__ unsigned int
__match32_all_sync(unsigned int mask, unsigned int value, int *pred);
inline __device__ unsigned long long
__match64_all_sync(unsigned int mask, unsigned long long value, int *pred);

inline __device__ char __ldg(const char *ptr);
inline __device__ short __ldg(const short *ptr);
inline __device__ int __ldg(const int *ptr);
inline __device__ long __ldg(const long *ptr);
inline __device__ long long __ldg(const long long *ptr);
inline __device__ unsigned char __ldg(const unsigned char *ptr);
inline __device__ signed char __ldg(const signed char *ptr);
inline __device__ unsigned short __ldg(const unsigned short *ptr);
inline __device__ unsigned int __ldg(const unsigned int *ptr);
inline __device__ unsigned long __ldg(const unsigned long *ptr);
inline __device__ unsigned long long __ldg(const unsigned long long *ptr);
inline __device__ float __ldg(const float *ptr);
inline __device__ double __ldg(const double *ptr);

inline __device__ unsigned __funnelshift_l(unsigned low32, unsigned high32,
                                           unsigned shiftWidth);
inline __device__ unsigned __funnelshift_lc(unsigned low32, unsigned high32,
                                            unsigned shiftWidth);
inline __device__ unsigned __funnelshift_r(unsigned low32, unsigned high32,
                                           unsigned shiftWidth);
inline __device__ unsigned __funnelshift_rc(unsigned low32, unsigned high32,
                                            unsigned shiftWidth);

extern "C" {
__device__ inline size_t __nv_cvta_generic_to_global_impl(const void *__ptr);
__device__ inline size_t __nv_cvta_generic_to_shared_impl(const void *__ptr);
__device__ inline size_t __nv_cvta_generic_to_constant_impl(const void *__ptr);
__device__ inline size_t __nv_cvta_generic_to_local_impl(const void *__ptr);
__device__ inline void *__nv_cvta_global_to_generic_impl(size_t __ptr);
__device__ inline void *__nv_cvta_shared_to_generic_impl(size_t __ptr);
__device__ inline void *__nv_cvta_constant_to_generic_impl(size_t __ptr);
__device__ inline void *__nv_cvta_local_to_generic_impl(size_t __ptr);
__device__ inline uint32_t __nvvm_get_smem_pointer(void *__ptr);
}

//* define complex operations
__DEVICE__ double _Complex __muldc3(double __a, double __b, double __c,
                                    double __d);
__DEVICE__ float _Complex __mulsc3(float __a, float __b, float __c, float __d);
__DEVICE__ double _Complex __divdc3(double __a, double __b, double __c,
                                    double __d);
__DEVICE__ float _Complex __divsc3(float __a, float __b, float __c, float __d);

//* define undocumented kernel access function
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                size_t sharedMem = 0,
                                                void *stream = 0);

//* define device functions
__DEVICE__ int __all(int __a);
__DEVICE__ int __any(int __a);
__DEVICE__ unsigned int __ballot(int __a);
__DEVICE__ unsigned int __brev(unsigned int __a);
__DEVICE__ unsigned long long __brevll(unsigned long long __a);
__DEVICE__ void __brkpt();
__DEVICE__ void __brkpt(int __a);

__DEVICE__ unsigned int __byte_perm(unsigned int __a, unsigned int __b,
                                    unsigned int __c);
__DEVICE__ int __clz(int __a);
__DEVICE__ int __clzll(long long __a);
__DEVICE__ float __cosf(float __a);
__DEVICE__ double __dAtomicAdd(double *__p, double __v);
__DEVICE__ double __dAtomicAdd_block(double *__p, double __v);
__DEVICE__ double __dAtomicAdd_system(double *__p, double __v);
__DEVICE__ double __dadd_rd(double __a, double __b);
__DEVICE__ double __dadd_rn(double __a, double __b);
__DEVICE__ double __dadd_ru(double __a, double __b);
__DEVICE__ double __dadd_rz(double __a, double __b);
__DEVICE__ double __ddiv_rd(double __a, double __b);
__DEVICE__ double __ddiv_rn(double __a, double __b);
__DEVICE__ double __ddiv_ru(double __a, double __b);
__DEVICE__ double __ddiv_rz(double __a, double __b);
__DEVICE__ double __dmul_rd(double __a, double __b);
__DEVICE__ double __dmul_rn(double __a, double __b);
__DEVICE__ double __dmul_ru(double __a, double __b);
__DEVICE__ double __dmul_rz(double __a, double __b);
__DEVICE__ float __double2float_rd(double __a);
__DEVICE__ float __double2float_rn(double __a);
__DEVICE__ float __double2float_ru(double __a);
__DEVICE__ float __double2float_rz(double __a);
__DEVICE__ int __double2hiint(double __a);
__DEVICE__ int __double2int_rd(double __a);
__DEVICE__ int __double2int_rn(double __a);
__DEVICE__ int __double2int_ru(double __a);
__DEVICE__ int __double2int_rz(double __a);
__DEVICE__ long long __double2ll_rd(double __a);
__DEVICE__ long long __double2ll_rn(double __a);
__DEVICE__ long long __double2ll_ru(double __a);
__DEVICE__ long long __double2ll_rz(double __a);
__DEVICE__ int __double2loint(double __a);
__DEVICE__ unsigned int __double2uint_rd(double __a);
__DEVICE__ unsigned int __double2uint_rn(double __a);
__DEVICE__ unsigned int __double2uint_ru(double __a);
__DEVICE__ unsigned int __double2uint_rz(double __a);
__DEVICE__ unsigned long long __double2ull_rd(double __a);
__DEVICE__ unsigned long long __double2ull_rn(double __a);
__DEVICE__ unsigned long long __double2ull_ru(double __a);
__DEVICE__ unsigned long long __double2ull_rz(double __a);
__DEVICE__ long long __double_as_longlong(double __a);
__DEVICE__ double __drcp_rd(double __a);
__DEVICE__ double __drcp_rn(double __a);
__DEVICE__ double __drcp_ru(double __a);
__DEVICE__ double __drcp_rz(double __a);
__DEVICE__ double __dsqrt_rd(double __a);
__DEVICE__ double __dsqrt_rn(double __a);
__DEVICE__ double __dsqrt_ru(double __a);
__DEVICE__ double __dsqrt_rz(double __a);
__DEVICE__ double __dsub_rd(double __a, double __b);
__DEVICE__ double __dsub_rn(double __a, double __b);
__DEVICE__ double __dsub_ru(double __a, double __b);
__DEVICE__ double __dsub_rz(double __a, double __b);
__DEVICE__ float __exp10f(float __a);
__DEVICE__ float __expf(float __a);
__DEVICE__ float __fAtomicAdd(float *__p, float __v);
__DEVICE__ float __fAtomicAdd_block(float *__p, float __v);
__DEVICE__ float __fAtomicAdd_system(float *__p, float __v);
__DEVICE__ float __fAtomicExch(float *__p, float __v);
__DEVICE__ float __fAtomicExch_block(float *__p, float __v);
__DEVICE__ float __fAtomicExch_system(float *__p, float __v);
__DEVICE__ float __fadd_rd(float __a, float __b);
__DEVICE__ float __fadd_rn(float __a, float __b);
__DEVICE__ float __fadd_ru(float __a, float __b);
__DEVICE__ float __fadd_rz(float __a, float __b);
__DEVICE__ float __fdiv_rd(float __a, float __b);
__DEVICE__ float __fdiv_rn(float __a, float __b);
__DEVICE__ float __fdiv_ru(float __a, float __b);
__DEVICE__ float __fdiv_rz(float __a, float __b);
__DEVICE__ float __fdividef(float __a, float __b);
__DEVICE__ int __ffs(int __a);
__DEVICE__ int __ffsll(long long __a);
__DEVICE__ int __finite(double __a);
__DEVICE__ int __finitef(float __a);
#ifdef _MSC_VER
__DEVICE__ int __finitel(long double __a);
#endif
__DEVICE__ int __float2int_rd(float __a);
__DEVICE__ int __float2int_rn(float __a);
__DEVICE__ int __float2int_ru(float __a);
__DEVICE__ int __float2int_rz(float __a);
__DEVICE__ long long __float2ll_rd(float __a);
__DEVICE__ long long __float2ll_rn(float __a);
__DEVICE__ long long __float2ll_ru(float __a);
__DEVICE__ long long __float2ll_rz(float __a);
__DEVICE__ unsigned int __float2uint_rd(float __a);
__DEVICE__ unsigned int __float2uint_rn(float __a);
__DEVICE__ unsigned int __float2uint_ru(float __a);
__DEVICE__ unsigned int __float2uint_rz(float __a);
__DEVICE__ unsigned long long __float2ull_rd(float __a);
__DEVICE__ unsigned long long __float2ull_rn(float __a);
__DEVICE__ unsigned long long __float2ull_ru(float __a);
__DEVICE__ unsigned long long __float2ull_rz(float __a);
__DEVICE__ int __float_as_int(float __a);
__DEVICE__ unsigned int __float_as_uint(float __a);
__DEVICE__ double __fma_rd(double __a, double __b, double __c);
__DEVICE__ double __fma_rn(double __a, double __b, double __c);
__DEVICE__ double __fma_ru(double __a, double __b, double __c);
__DEVICE__ double __fma_rz(double __a, double __b, double __c);
__DEVICE__ float __fmaf_ieee_rd(float __a, float __b, float __c);
__DEVICE__ float __fmaf_ieee_rn(float __a, float __b, float __c);
__DEVICE__ float __fmaf_ieee_ru(float __a, float __b, float __c);
__DEVICE__ float __fmaf_ieee_rz(float __a, float __b, float __c);
__DEVICE__ float __fmaf_rd(float __a, float __b, float __c);
__DEVICE__ float __fmaf_rn(float __a, float __b, float __c);
__DEVICE__ float __fmaf_ru(float __a, float __b, float __c);
__DEVICE__ float __fmaf_rz(float __a, float __b, float __c);
__DEVICE__ float __fmul_rd(float __a, float __b);
__DEVICE__ float __fmul_rn(float __a, float __b);
__DEVICE__ float __fmul_ru(float __a, float __b);
__DEVICE__ float __fmul_rz(float __a, float __b);
__DEVICE__ float __frcp_rd(float __a);
__DEVICE__ float __frcp_rn(float __a);
__DEVICE__ float __frcp_ru(float __a);
__DEVICE__ float __frcp_rz(float __a);
__DEVICE__ float __frsqrt_rn(float __a);
__DEVICE__ float __fsqrt_rd(float __a);
__DEVICE__ float __fsqrt_rn(float __a);
__DEVICE__ float __fsqrt_ru(float __a);
__DEVICE__ float __fsqrt_rz(float __a);
__DEVICE__ float __fsub_rd(float __a, float __b);
__DEVICE__ float __fsub_rn(float __a, float __b);
__DEVICE__ float __fsub_ru(float __a, float __b);
__DEVICE__ float __fsub_rz(float __a, float __b);
__DEVICE__ int __hadd(int __a, int __b);
__DEVICE__ double __hiloint2double(int __a, int __b);
__DEVICE__ int __iAtomicAdd(int *__p, int __v);
__DEVICE__ int __iAtomicAdd_block(int *__p, int __v);
__DEVICE__ int __iAtomicAdd_system(int *__p, int __v);
__DEVICE__ int __iAtomicAnd(int *__p, int __v);
__DEVICE__ int __iAtomicAnd_block(int *__p, int __v);
__DEVICE__ int __iAtomicAnd_system(int *__p, int __v);
__DEVICE__ int __iAtomicCAS(int *__p, int __cmp, int __v);
__DEVICE__ int __iAtomicCAS_block(int *__p, int __cmp, int __v);
__DEVICE__ int __iAtomicCAS_system(int *__p, int __cmp, int __v);
__DEVICE__ int __iAtomicExch(int *__p, int __v);
__DEVICE__ int __iAtomicExch_block(int *__p, int __v);
__DEVICE__ int __iAtomicExch_system(int *__p, int __v);
__DEVICE__ int __iAtomicMax(int *__p, int __v);
__DEVICE__ int __iAtomicMax_block(int *__p, int __v);
__DEVICE__ int __iAtomicMax_system(int *__p, int __v);
__DEVICE__ int __iAtomicMin(int *__p, int __v);
__DEVICE__ int __iAtomicMin_block(int *__p, int __v);
__DEVICE__ int __iAtomicMin_system(int *__p, int __v);
__DEVICE__ int __iAtomicOr(int *__p, int __v);
__DEVICE__ int __iAtomicOr_block(int *__p, int __v);
__DEVICE__ int __iAtomicOr_system(int *__p, int __v);
__DEVICE__ int __iAtomicXor(int *__p, int __v);
__DEVICE__ int __iAtomicXor_block(int *__p, int __v);
__DEVICE__ int __iAtomicXor_system(int *__p, int __v);
__DEVICE__ long long __illAtomicMax(long long *__p, long long __v);
__DEVICE__ long long __illAtomicMax_block(long long *__p, long long __v);
__DEVICE__ long long __illAtomicMax_system(long long *__p, long long __v);
__DEVICE__ long long __illAtomicMin(long long *__p, long long __v);
__DEVICE__ long long __illAtomicMin_block(long long *__p, long long __v);
__DEVICE__ long long __illAtomicMin_system(long long *__p, long long __v);
__DEVICE__ double __int2double_rn(int __a);
__DEVICE__ float __int2float_rd(int __a);
__DEVICE__ float __int2float_rn(int __a);
__DEVICE__ float __int2float_ru(int __a);
__DEVICE__ float __int2float_rz(int __a);
__DEVICE__ float __int_as_float(int __a);
__DEVICE__ int __isfinited(double __a);
__DEVICE__ int __isinf(double __a);
__DEVICE__ int __isinff(float __a);
#ifdef _MSC_VER
__DEVICE__ int __isinfl(long double __a);
#endif
__DEVICE__ int __isnan(double __a);
__DEVICE__ int __isnanf(float __a);
#ifdef _MSC_VER
__DEVICE__ int __isnanl(long double __a);
#endif
__DEVICE__ double __ll2double_rd(long long __a);
__DEVICE__ double __ll2double_rn(long long __a);
__DEVICE__ double __ll2double_ru(long long __a);
__DEVICE__ double __ll2double_rz(long long __a);
__DEVICE__ float __ll2float_rd(long long __a);
__DEVICE__ float __ll2float_rn(long long __a);
__DEVICE__ float __ll2float_ru(long long __a);
__DEVICE__ float __ll2float_rz(long long __a);
__DEVICE__ long long __llAtomicAnd(long long *__p, long long __v);
__DEVICE__ long long __llAtomicAnd_block(long long *__p, long long __v);
__DEVICE__ long long __llAtomicAnd_system(long long *__p, long long __v);
__DEVICE__ long long __llAtomicOr(long long *__p, long long __v);
__DEVICE__ long long __llAtomicOr_block(long long *__p, long long __v);
__DEVICE__ long long __llAtomicOr_system(long long *__p, long long __v);
__DEVICE__ long long __llAtomicXor(long long *__p, long long __v);
__DEVICE__ long long __llAtomicXor_block(long long *__p, long long __v);
__DEVICE__ long long __llAtomicXor_system(long long *__p, long long __v);
__DEVICE__ float __log10f(float __a);
__DEVICE__ float __log2f(float __a);
__DEVICE__ float __logf(float __a);
__DEVICE__ double __longlong_as_double(long long __a);
__DEVICE__ int __mul24(int __a, int __b);
__DEVICE__ long long __mul64hi(long long __a, long long __b);
__DEVICE__ int __mulhi(int __a, int __b);
__DEVICE__ unsigned int __pm0(void);
__DEVICE__ unsigned int __pm1(void);
__DEVICE__ unsigned int __pm2(void);
__DEVICE__ unsigned int __pm3(void);
__DEVICE__ int __popc(int __a);
__DEVICE__ int __popcll(long long __a);
__DEVICE__ float __powf(float __a, float __b);

__DEVICE__ int __rhadd(int __a, int __b);
__DEVICE__ unsigned int __sad(int __a, int __b, unsigned int __c);
__DEVICE__ float __saturatef(float __a);
__DEVICE__ int __signbitd(double __a);
__DEVICE__ int __signbitf(float __a);
__DEVICE__ void __sincosf(float __a, float *__s, float *__c);
__DEVICE__ float __sinf(float __a);
__DEVICE__ int __syncthreads_and(int __a);
__DEVICE__ int __syncthreads_count(int __a);
__DEVICE__ int __syncthreads_or(int __a);
__DEVICE__ float __tanf(float __a);
__DEVICE__ void __threadfence(void);
__DEVICE__ void __threadfence_block(void);
__DEVICE__ void __threadfence_system(void);
__DEVICE__ void __trap(void);
__DEVICE__ unsigned int __uAtomicAdd(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicAdd_block(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicAdd_system(unsigned int *__p,
                                            unsigned int __v);
__DEVICE__ unsigned int __uAtomicAnd(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicAnd_block(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicAnd_system(unsigned int *__p,
                                            unsigned int __v);
__DEVICE__ unsigned int __uAtomicCAS(unsigned int *__p, unsigned int __cmp,
                                     unsigned int __v);
__DEVICE__ unsigned int
__uAtomicCAS_block(unsigned int *__p, unsigned int __cmp, unsigned int __v);
__DEVICE__ unsigned int
__uAtomicCAS_system(unsigned int *__p, unsigned int __cmp, unsigned int __v);
__DEVICE__ unsigned int __uAtomicDec(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicDec_block(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicDec_system(unsigned int *__p,
                                            unsigned int __v);
__DEVICE__ unsigned int __uAtomicExch(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicExch_block(unsigned int *__p,
                                            unsigned int __v);
__DEVICE__ unsigned int __uAtomicExch_system(unsigned int *__p,
                                             unsigned int __v);
__DEVICE__ unsigned int __uAtomicInc(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicInc_block(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicInc_system(unsigned int *__p,
                                            unsigned int __v);
__DEVICE__ unsigned int __uAtomicMax(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicMax_block(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicMax_system(unsigned int *__p,
                                            unsigned int __v);
__DEVICE__ unsigned int __uAtomicMin(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicMin_block(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicMin_system(unsigned int *__p,
                                            unsigned int __v);
__DEVICE__ unsigned int __uAtomicOr(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicOr_block(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicOr_system(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicXor(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicXor_block(unsigned int *__p, unsigned int __v);
__DEVICE__ unsigned int __uAtomicXor_system(unsigned int *__p,
                                            unsigned int __v);
__DEVICE__ unsigned int __uhadd(unsigned int __a, unsigned int __b);
__DEVICE__ double __uint2double_rn(unsigned int __a);
__DEVICE__ float __uint2float_rd(unsigned int __a);
__DEVICE__ float __uint2float_rn(unsigned int __a);
__DEVICE__ float __uint2float_ru(unsigned int __a);
__DEVICE__ float __uint2float_rz(unsigned int __a);
__DEVICE__ float __uint_as_float(unsigned int __a); //
__DEVICE__ double __ull2double_rd(unsigned long long __a);
__DEVICE__ double __ull2double_rn(unsigned long long __a);
__DEVICE__ double __ull2double_ru(unsigned long long __a);
__DEVICE__ double __ull2double_rz(unsigned long long __a);
__DEVICE__ float __ull2float_rd(unsigned long long __a);
__DEVICE__ float __ull2float_rn(unsigned long long __a);
__DEVICE__ float __ull2float_ru(unsigned long long __a);
__DEVICE__ float __ull2float_rz(unsigned long long __a);
__DEVICE__ unsigned long long __ullAtomicAdd(unsigned long long *__p,
                                             unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicAdd_block(unsigned long long *__p,
                                                   unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicAdd_system(unsigned long long *__p,
                                                    unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicAnd(unsigned long long *__p,
                                             unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicAnd_block(unsigned long long *__p,
                                                   unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicAnd_system(unsigned long long *__p,
                                                    unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicCAS(unsigned long long *__p,
                                             unsigned long long __cmp,
                                             unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicCAS_block(unsigned long long *__p,
                                                   unsigned long long __cmp,
                                                   unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicCAS_system(unsigned long long *__p,
                                                    unsigned long long __cmp,
                                                    unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicExch(unsigned long long *__p,
                                              unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicExch_block(unsigned long long *__p,
                                                    unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicExch_system(unsigned long long *__p,
                                                     unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicMax(unsigned long long *__p,
                                             unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicMax_block(unsigned long long *__p,
                                                   unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicMax_system(unsigned long long *__p,
                                                    unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicMin(unsigned long long *__p,
                                             unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicMin_block(unsigned long long *__p,
                                                   unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicMin_system(unsigned long long *__p,
                                                    unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicOr(unsigned long long *__p,
                                            unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicOr_block(unsigned long long *__p,
                                                  unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicOr_system(unsigned long long *__p,
                                                   unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicXor(unsigned long long *__p,
                                             unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicXor_block(unsigned long long *__p,
                                                   unsigned long long __v);
__DEVICE__ unsigned long long __ullAtomicXor_system(unsigned long long *__p,
                                                    unsigned long long __v);
__DEVICE__ unsigned int __umul24(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned long long __umul64hi(unsigned long long __a,
                                         unsigned long long __b);
__DEVICE__ unsigned int __umulhi(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __urhadd(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __usad(unsigned int __a, unsigned int __b,
                               unsigned int __c);

__DEVICE__ unsigned int __vabs2(unsigned int __a);
__DEVICE__ unsigned int __vabs4(unsigned int __a);
__DEVICE__ unsigned int __vabsdiffs2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vabsdiffs4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vabsdiffu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vabsdiffu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vabsss2(unsigned int __a);
__DEVICE__ unsigned int __vabsss4(unsigned int __a);
__DEVICE__ unsigned int __vadd2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vadd4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vaddss2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vaddss4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vaddus2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vaddus4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vavgs2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vavgs4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vavgu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vavgu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpeq2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpeq4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpges2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpges4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpgeu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpgeu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpgts2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpgts4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpgtu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpgtu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmples2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmples4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpleu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpleu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmplts2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmplts4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpltu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpltu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpne2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vcmpne4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vhaddu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vhaddu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vmaxs2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vmaxs4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vmaxu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vmaxu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vmins2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vmins4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vminu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vminu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vneg2(unsigned int __a);
__DEVICE__ unsigned int __vneg4(unsigned int __a);
__DEVICE__ unsigned int __vnegss2(unsigned int __a);
__DEVICE__ unsigned int __vnegss4(unsigned int __a);
__DEVICE__ unsigned int __vsads2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsads4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsadu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsadu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vseteq2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vseteq4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetges2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetges4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetgeu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetgeu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetgts2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetgts4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetgtu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetgtu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetles2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetles4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetleu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetleu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetlts2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetlts4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetltu2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetltu4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetne2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsetne4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsub2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsub4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsubss2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsubss4(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsubus2(unsigned int __a, unsigned int __b);
__DEVICE__ unsigned int __vsubus4(unsigned int __a, unsigned int __b);

__DEVICE__ /* clock_t= */ int clock();
__DEVICE__ long long clock64();

__DEVICE__ void *memcpy(void *__a, const void *__b, size_t __c);
__DEVICE__ void *memset(void *__a, int __b, size_t __c);

#undef __DEVICE__

#endif // __CUDACC__

#endif // CUDAMASU_H_
