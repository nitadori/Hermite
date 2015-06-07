#ifndef __AVX__
#error
#endif

typedef double v2df __attribute__((vector_size(16)));
typedef double v4df __attribute__((vector_size(32)));

static inline double v4df_elem0(const v4df vec){
	return __builtin_ia32_vec_ext_v2df(
			__builtin_ia32_vextractf128_pd256(vec, 0),
			0);
}

static inline double v4df_elem1(const v4df vec){
	return __builtin_ia32_vec_ext_v2df(
			__builtin_ia32_vextractf128_pd256(vec, 0),
			1);
}

static inline double v4df_elem2(const v4df vec){
	return __builtin_ia32_vec_ext_v2df(
			__builtin_ia32_vextractf128_pd256(vec, 1),
			0);
}
static inline double v4df_elem3(const v4df vec){
	return __builtin_ia32_vec_ext_v2df(
			__builtin_ia32_vextractf128_pd256(vec, 1),
			1);
}

static inline int print_v4df(
		const v4df vec,
		FILE *fp = stdout,
		const char *fmt = "{%f, %f, %f, %f}\n")
{
	return fprintf(fp, fmt, 
			v4df_elem0(vec),
			v4df_elem1(vec),
			v4df_elem2(vec),
			v4df_elem3(vec));
}

#define REP4(x) {x, x, x, x}

static inline v4df v4df_rsqrt(const v4df x){
#if 0
	return (v4df){1.0, 1.0, 1.0, 1.0} / __builtin_ia32_sqrtpd256(x);
#else
	const v4df y0 = __builtin_ia32_cvtps2pd256(
	                    __builtin_ia32_rsqrtps(
	                      __builtin_ia32_cvtpd2ps256(x)));
	const v4df c0 = REP4(1.0);
	const v4df c1 = REP4(0.5);
	const v4df c2 = REP4(3./8.);
	const v4df c3 = REP4(5./16.);
	const v4df c4 = REP4(35./128.);

	const v4df h  = c0 - x*y0*y0;
#if 1
	// const v4df y1 = y0 * (c0 + h * (c1 + h * (c2 + h * (c3 + h * (c4)))));
	const v4df y1 = y0  + (y0*h) * (c1 + h * (c2 + h * (c3 + h * (c4))));
#else
	// latency orient
	const v4df h2 = h*h;
	const v4df a  = c1 + h * c2;
	const v4df b  = c3 + h * c4;
	const v4df p  = h*(a + h2*b);
	const v4df y1 = y0 + p*y0;
#endif
	return y1;
#endif
}

static inline dvec3 v4df_to_dvec3(const v4df vec){
	return dvec3(v4df_elem0(vec), v4df_elem1(vec), v4df_elem2(vec));
}

struct v4df_bcast{
	v4df e0, e1, e2, e3;
	v4df_bcast(const v4df *ptr){
#ifdef __AVX2__
		v4df tmp = *ptr;
		e0 =  __builtin_ia32_permdf256(tmp, 0x00);
		e1 =  __builtin_ia32_permdf256(tmp, 0x55);
		e2 =  __builtin_ia32_permdf256(tmp, 0xaa);
		e3 =  __builtin_ia32_permdf256(tmp, 0xff);
#else
		const v4df tmp0 = __builtin_ia32_vbroadcastf128_pd256((const v2df*)(ptr) + 0);
		const v4df tmp1 = __builtin_ia32_vbroadcastf128_pd256((const v2df*)(ptr) + 1);
		e0 = __builtin_ia32_vpermilpd256(tmp0, 0x0);
		e1 = __builtin_ia32_vpermilpd256(tmp0, 0xf);
		e2 = __builtin_ia32_vpermilpd256(tmp1, 0x0);
		e3 = __builtin_ia32_vpermilpd256(tmp1, 0xf);
#endif

	}
};

struct v4df_transpose{
	v4df c0,c1, c2, c3;
	v4df_transpose(const v4df r0, const v4df r1, const v4df r2, const v4df r3){
		const v4df tmp0 = __builtin_ia32_unpcklpd256(r0, r1); // | r12 | r02 | r10 | r00 |
		const v4df tmp1 = __builtin_ia32_unpckhpd256(r0, r1); // | r13 | r03 | r11 | r01 |
		const v4df tmp2 = __builtin_ia32_unpcklpd256(r2, r3); // | r32 | r22 | r30 | r20 |
		const v4df tmp3 = __builtin_ia32_unpckhpd256(r2, r3); // | r33 | r23 | r31 | r21 |
		c0 = __builtin_ia32_vperm2f128_pd256(tmp0, tmp2, (0)+(2<<4));
		c1 = __builtin_ia32_vperm2f128_pd256(tmp1, tmp3, (0)+(2<<4));
		c2 = __builtin_ia32_vperm2f128_pd256(tmp0, tmp2, (1)+(3<<4));
		c3 = __builtin_ia32_vperm2f128_pd256(tmp1, tmp3, (1)+(3<<4));
	}
};

// from {ax, ay, az, --} and {bx, by, bz, --}
// returns {ax, ay, ax, bx}
static inline v4df v4df_vecalign1(const v4df a, const v4df b){
	const v4df bxyxy = __builtin_ia32_vperm2f128_pd256(b, b, 0);
	const v4df bxxxx = __builtin_ia32_vpermilpd256(bxyxy, 0);
	return __builtin_ia32_blendpd256(a, bxxxx, 8); // 0b1000
}
// returns {ay, az, bx, by}
static inline v4df v4df_vecalign2(const v4df a, const v4df b){
	const v4df az__bxby = __builtin_ia32_vperm2f128_pd256(a, b, (1)+(2<<4));
	const v4df azaybxby = __builtin_ia32_blendpd256(az__bxby, a, 2); // 0b0010
	return __builtin_ia32_vpermilpd256(azaybxby, 9); // 0b1001
}
// returns {az, bx, by, bz}
static inline v4df v4df_vecalign3(const v4df a, const v4df b){
	// {az, --, bz, __} -> {az, __, __, bz}
	// {bx, by, bx, by} -> {bx, bx, by, by}
	const v4df az__bz__ = __builtin_ia32_vperm2f128_pd256(a, b, (1)+(3<<4));
	const v4df az____bz = __builtin_ia32_vpermilpd256(az__bz__, 0);
	const v4df bxbybxby = __builtin_ia32_vperm2f128_pd256(b, b, 0);
	const v4df bxbxbyby = __builtin_ia32_vpermilpd256(bxbybxby, 12); // 0b1100
	return __builtin_ia32_blendpd256(az____bz, bxbxbyby, 6); // 0b0110
}
