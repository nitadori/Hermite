#pragma once
#include <micvec.h>

static inline __m512d loadbcast2f256(const void *ptr, const int hint=0){
	return _mm512_extload_pd(ptr, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, hint);
}

#if 1
static inline __m512d loadu_pd(const void *ptr){
	__m512d ret = _mm512_undefined_pd();
	ret = _mm512_loadunpacklo_pd(ret, (const double *)ptr + 0);
	ret = _mm512_loadunpackhi_pd(ret, (const double *)ptr + 8);
	return ret;
}
static inline void storeu_pd(void *ptr, __m512d vec){
	_mm512_packstorelo_pd((double *)ptr + 0, vec);
	_mm512_packstorehi_pd((double *)ptr + 8, vec);
}
static inline void storeu_pd256(void *ptr, __m512d vec){
	_mm512_mask_packstorelo_pd((double *)ptr + 0, 0x0f, vec);
}
#endif

static inline __m512d rsqrt_pd_x3(
		__m512d x, __m512d _one, __m512d _1_ov_2, __m512d _3_ov_8)
{
	F64vec8 y0 =  _mm512_cvtpslo_pd(
	                _mm512_rsqrt23_ps(
					   	_mm512_cvtpd_pslo(x)));
	F64vec8 h0 = -(x*(y0*y0) - _one);
	F64vec8 tmp = _one + (h0 * (_1_ov_2 + h0 * _3_ov_8));
	return tmp * y0;
}

static inline __m512d rsqrt_pd_x3_v2(
		__m512d x, __m512d _one, __m512d _7_ov_3, __m512d _3_ov_8)
{
	F64vec8 y0 =  _mm512_cvtpslo_pd(
	                _mm512_rsqrt23_ps(
					   	_mm512_cvtpd_pslo(x)));
	F64vec8 y2 = y0*y0;
	F64vec8 h0 = _one - x*y2;
	F64vec8 ha = _7_ov_3 - x*y2;
	F64vec8 p  = (_3_ov_8 * h0) * ha;
	return (y0 + p*y0);
}

static inline F64vec8 permute4f128(F64vec8 inp, _MM_PERM_ENUM perm){
	return (F64vec8)_mm512_permute4f128_epi32(
			_mm512_castpd_si512(inp), perm);
}

static inline void pack_2f256(F64vec8 &x, F64vec8 &y){
	// x := {xl, yl}
	// y := {xh, yh}
	F64vec8 xx = permute4f128(x, _MM_PERM_DCDC);
	F64vec8 yy = permute4f128(y, _MM_PERM_BABA);
	x = _mm512_mask_blend_pd(0xf0, x, yy);
	y = _mm512_mask_blend_pd(0xf,  y, xx);
}

static inline F64vec8 hadd_2f256(F64vec8 x, F64vec8 y){
	// x := {xl, yl}
	// y := {xh, yh}
	// ret := {xl+xh, yl+yh}
	x += permute4f128(x, _MM_PERM_DCDC);
	y += permute4f128(y, _MM_PERM_BABA);
	return _mm512_mask_blend_pd(0xf0, x, y);
}

static inline void transpose_4zmm_pd(F64vec8 &v0, F64vec8 &v1, F64vec8 &v2, F64vec8 &v3){
	F64vec8 ccaa_1100 = _mm512_mask_blend_pd(0xaa, v0, v1.cdab());
	F64vec8 ccaa_3322 = _mm512_mask_blend_pd(0xaa, v2, v3.cdab());
	F64vec8 ddbb_1100 = _mm512_mask_blend_pd(0x55, v1, v0.cdab());
	F64vec8 ddbb_3322 = _mm512_mask_blend_pd(0x55, v3, v2.cdab());

	F64vec8 aaaa = _mm512_mask_blend_pd(0xcc, ccaa_1100, ccaa_3322.badc());
	F64vec8 bbbb = _mm512_mask_blend_pd(0xcc, ddbb_1100, ddbb_3322.badc());
	F64vec8 cccc = _mm512_mask_blend_pd(0x33, ccaa_3322, ccaa_1100.badc());
	F64vec8 dddd = _mm512_mask_blend_pd(0x33, ddbb_3322, ddbb_1100.badc());

	v0 = aaaa;
	v1 = bbbb;
	v2 = cccc;
	v3 = dddd;
}

static inline void triple_unpack1(F64vec8 &xv, F64vec8 &ax, F64vec8 &va){
	F64vec8 xx = _mm512_mask_blend_pd(0xf0, xv, ax);
	F64vec8 aa = _mm512_mask_blend_pd(0xf0, ax, va);
	F64vec8 vl = permute4f128(xv, _MM_PERM_DCDC);
	F64vec8 vh = permute4f128(va, _MM_PERM_BABA);
	F64vec8 vv = _mm512_mask_blend_pd(0xf0, vl, vh);

	xv = xx;
	ax = vv;
	va = aa;
}

static inline void triple_unpack2(F64vec8 &a1a0, F64vec8 &j1j0, F64vec8 &s1s0){
	F64vec8 j0j1 = permute4f128(j1j0, _MM_PERM_BADC);

	F64vec8 j0a0 = _mm512_mask_blend_pd(0xf0, a1a0, j0j1);
	F64vec8 a1s0 = _mm512_mask_blend_pd(0xf0, s1s0, a1a0);
	F64vec8 s1j1 = _mm512_mask_blend_pd(0xf0, j0j1, s1s0);

	a1a0 = j0a0;
	j1j0 = a1s0;
	s1s0 = s1j1;
}

static inline int aligned_division(
		const int n, 
		const int tid, 
		const int nth, 
		const int align)
{
	assert(0 == n%align);
	const int na = n/align;
	return align * ((na*tid)/nth);
}
