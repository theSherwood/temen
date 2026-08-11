#ifndef __SVM_XMMINTRIN_H
#define __SVM_XMMINTRIN_H
// Minimal SSE (packed single-precision, "_ps") surface for the playground. `__m128` is a chibicc
// builtin type — a 16-byte SIMD vector backed by the IR `v128` value (four `float` lanes). These
// prototypes are recognized as intrinsics by codegen and lowered directly to `f32x4`/`v128` ops;
// no function bodies are linked. Scope is the teaching set (arithmetic, broadcast/construct,
// load/store) — add more `_mm_*` as lessons demand them.

__m128 _mm_add_ps(__m128 a, __m128 b);
__m128 _mm_sub_ps(__m128 a, __m128 b);
__m128 _mm_mul_ps(__m128 a, __m128 b);
__m128 _mm_min_ps(__m128 a, __m128 b);
__m128 _mm_max_ps(__m128 a, __m128 b);

__m128 _mm_set1_ps(float x);
__m128 _mm_setzero_ps(void);
__m128 _mm_set_ps(float e3, float e2, float e1, float e0);

__m128 _mm_load_ps(const float *p);
__m128 _mm_loadu_ps(const float *p);
void _mm_store_ps(float *p, __m128 v);
void _mm_storeu_ps(float *p, __m128 v);

#endif // __SVM_XMMINTRIN_H
