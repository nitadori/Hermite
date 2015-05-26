#include <cstdio>
#include <mxintrin.h>

struct v4r8;
struct v4_mask{
	typedef __builtin_v4r8 _v4r8;
	_v4r8 mask;

	v4_mask(const v4_mask &src) : mask(src.mask) {}
	v4_mask operator=(const v4_mask src) {
	   	mask = src.mask;
		return *this;
	}

	v4_mask(const _v4r8 _val) : mask(_val) {}
	v4_mask(const int m0, const int m1, const int m2, const int m3)
		: mask(_fjsp_set_v4i8(
					m3 ? -1L : 0, 
					m2 ? -1L : 0, 
					m1 ? -1L : 0, 
					m0 ? -1L : 0) ){}
	v4r8 select(const v4r8 a, const v4r8 b) const;
};

struct v4r8{
	static void simd_mode_4() { _fjsp_simd_mode_4(); }
	static void simd_mode_2() { _fjsp_simd_mode_2(); }

	typedef __builtin_v4r8 _v4r8;
	_v4r8 val;

	v4r8(const v4r8 &src) : val(src.val) {}
	v4r8 operator=(const v4r8 src) {
	   	val = src.val;
		return *this;
	}

	v4r8() : val(_fjsp_setzero_v4r8() ) {}
	v4r8(const _v4r8 _val) : val(_val) {}
	v4r8(const double a) : val(_fjsp_set_v4r8(a, a, a, a)) {}
	v4r8(const double a, const double b, const double c, const double d)
	   	: val(_fjsp_set_v4r8(d, c, b, a)) {}

	operator _v4r8() { return val; }

	static v4r8 load(const double *p){
		return v4r8( _fjsp_load_v4r8(p) );
	}
	// this version kills the compiler
	static v4r8 strideload(const double *p, const int stride){
		return v4r8( _fjsp_strideload_v4r8(p, stride) );
	}
	template <int stride>
	static v4r8 strideload(const double *p){
		return v4r8( _fjsp_strideload_v4r8(p, stride) );
	}
	static v4r8 broadcastload(const double *p){
		return v4r8( _fjsp_broadcastload_v4r8(p) );
	}
	void store(double *p) const {
		_fjsp_store_v4r8(p, val);
		// *(_v4r8 *)p = val;
	}
	// this version kills the compiler
	void stridestore(double *p, const int stride) const {
		_fjsp_stridestore_v4r8(p, val, stride);
	}
	template <int stride>
	void stridestore(double *p) const {
		_fjsp_stridestore_v4r8(p, val, stride);
	}
	void selstore(double *p, const int m0, const int m1, const int m2, const int m3) const {
		// true  : store
		// false : not store
		_fjsp_selstore_v4r8(p, val, 
				_fjsp_set_v4i8(
					m3 ? -1L : 0, 
					m2 ? -1L : 0, 
					m1 ? -1L : 0, 
					m0 ? -1L : 0));
	}
	void store(double *p, const v4_mask m) const {
		_fjsp_selstore_v4r8(p, val, m.mask);
	}

	double extract(const int pos) const {
		return _fjsp_extract_v4r8(val, pos);
	}

	int print(FILE *fp=stdout, const char * fmt="(%e, %e, %e, %e)\n") const {
		return fprintf(fp, fmt, extract(0), extract(1), extract(2), extract(3));
	}

	v4r8 eperm(const int e0, const int e1, const int e2, const int e3) const {
		return v4r8(_fjsp_eperm_v4(val, _fjsp_set_v4i4(e3, e2, e1, e0)));
	}

	v4r8 ecsl(const v4r8 rhs, const int ishift) const {
		// concat and shift : (val, rhs) << ishift
		return _fjsp_ecsl_v4(val, rhs.val, ishift);
	}
	v4r8 ecsl(const int ishift) const {
		// concat and shift : (val, rhs) << ishift
		return _fjsp_ecsl_v4(val, val, ishift);
	}

	v4r8 operator+(const v4r8 rhs) const {
		return v4r8(_fjsp_add_v4r8(val, rhs.val));
	}
	v4r8 operator-(const v4r8 rhs) const {
		return v4r8(_fjsp_sub_v4r8(val, rhs.val));
	}
	v4r8 operator*(const v4r8 rhs) const {
		return v4r8(_fjsp_mul_v4r8(val, rhs.val));
	}
	v4r8 operator+=(const v4r8 rhs) {
		return v4r8(val = _fjsp_add_v4r8(val, rhs.val));
	}
	v4r8 operator-=(const v4r8 rhs) {
		return v4r8(val = _fjsp_sub_v4r8(val, rhs.val));
	}
	v4r8 operator*=(const v4r8 rhs) {
		return v4r8(val = _fjsp_mul_v4r8(val, rhs.val));
	}

	v4_mask operator<(const v4r8 rhs) const {
		return v4_mask(_fjsp_cmplte_v4r8(val, rhs.val));
	}

	v4r8 rcpa()   const { return v4r8( _fjsp_rcpa_v4r8  (val) ); }
	v4r8 rsqrta() const { return v4r8( _fjsp_rsqrta_v4r8(val) ); }
	__attribute((always_inline))
	v4r8 rsqrta_x8() const {
		const v4r8 x(val);
		const v4r8 y0 = v4r8(__builtin_fj_rsqrta_v4r8(val));
		const v4r8 half(0.5);
		const v4r8 xhalf = x * half;
		const v4r8 h0half = half - y0*(y0*xhalf);
		const v4r8 y1 = y0 + y0*h0half;           // x2
		const v4r8 h1half = half - y1*(y1*xhalf);
		const v4r8 y2 = y1 + y1*h1half;           // x4
		const v4r8 h2half = half - y2*(y2*xhalf);
		const v4r8 y3 = y2 + y2*h2half;           // x8
		return y3; // 4-mul, 6-fma
	}

	static v4r8 nan4() { return v4r8(_fjsp_set_v4i8(-1, -1, -1, -1)); }

	static v4r8 select(const v4r8 a, const v4r8 b, const int m0, const int m1, const int m2, const int m3) {
		// true  : select a
		// false : selsct b 
		return v4r8(_fjsp_selmov_v4r8(
					a.val, 
					b.val, 
					_fjsp_set_v4i8(
						m3 ? -1L : 0, 
						m2 ? -1L : 0, 
						m1 ? -1L : 0, 
						m0 ? -1L : 0)));
	}

	__attribute((always_inline))
	static void transpose(v4r8 &a, v4r8 &b, v4r8 &c, v4r8 &d){ 
		// 6-fsel, 8-ecsl, 2-mask-reg
		const v4_mask m1010(1,0,1,0);
		const v4_mask m1100(1,1,0,0);
		v4r8 a0b0a2b2 = m1010.select(a,           b.ecsl(b,3));
		//                           a0a1a2ae3e   b3b0b1b2
		v4r8 a1b1a3b3 = m1010.select(a.ecsl(a,1), b          );
		//                           a1a2a3a0     b0b1b2b3
		v4r8 c2d2c0d0 = m1010.select(c.ecsl(c,2), d.ecsl(d,1));
		//                           c2c3c0c1     d1d2d3d0
		v4r8 c3d3c1d1 = m1010.select(c.ecsl(c,3), d.ecsl(d,2));
		//                           c3c0c1c2     d2d3d0d1
		v4r8 a0b0c0d0 = m1100.select(a0b0a2b2, c2d2c0d0);
		v4r8 a1b1c1d1 = m1100.select(a1b1a3b3, c3d3c1d1);
		v4r8 a2b2c2d2 = a0b0a2b2.ecsl(c2d2c0d0, 2);
		v4r8 a3b3c3d3 = a1b1a3b3.ecsl(c3d3c1d1, 2);

		a = a0b0c0d0;
		b = a1b1c1d1;
		c = a2b2c2d2;
		d = a3b3c3d3;
	}

	v4r8 swizzle_0101() const {
		return ecsl(*this, 2).ecsl(*this, 2);
	}
	v4r8 swizzle_2323() const {
		return ecsl(this->ecsl(*this, 2), 2);
	}
	v4r8 swizzle_0022() const {
		return v4r8( _fjsp_madd_cp_v4r8(val, v4r8(1.0).val, _fjsp_setzero_v4r8()) );
	}
	v4r8 swizzle_1133() const {
		return v4r8( _fjsp_madd_cp_sr1_v4r8(val, v4r8(1.0).val, _fjsp_setzero_v4r8()) );
	}
	static v4r8 const_1p0(){
		static const v4r8 c(1.0);
		return c;
	}
};

static inline v4r8 operator&(const v4r8 lhs, const v4_mask rhs){
	return v4r8(_fjsp_and_v4r8(lhs.val, rhs.mask));
}

inline v4r8 v4_mask::select(const v4r8 a, const v4r8 b) const {
	return v4r8(_fjsp_selmov_v4r8(a.val, b.val, this->mask));
}

#if 1
// swizzle types
struct v4r8_0022{
	typedef __builtin_v4r8 _v4r8;
	_v4r8 val;
	v4r8_0022(const v4r8 src) : val(src.val) {}
	v4r8 operator-(const v4r8 rhs) const {
		const v4r8 one(1.0);
		return v4r8( _fjsp_msub_cp_v4r8(val, one.val, rhs.val) );
	}
	friend v4r8 operator-(const v4r8 lhs, const v4r8_0022 rhs){
		const v4r8 one(1.0);
		return v4r8( _fjsp_nmsub_cp_v4r8(rhs.val, one.val, lhs.val) );
	}
	v4r8 operator*(const v4r8 rhs) const {
		const _v4r8 zero = _fjsp_setzero_v4r8();
		return v4r8( _fjsp_madd_cp_v4r8(val, rhs.val, zero) );
	}
	friend v4r8 operator*(const v4r8 lhs, const v4r8_0022 rhs){
		const _v4r8 zero = _fjsp_setzero_v4r8();
		return v4r8( _fjsp_madd_cp_v4r8(rhs.val, lhs.val, zero) );
	}
	v4r8 madd(const v4r8 b, const v4r8 c) const {
		return v4r8( _fjsp_madd_cp_v4r8(val, b.val, c.val) );
	}
	v4r8 nmsub(const v4r8 b, const v4r8 c) const {
		return v4r8( _fjsp_nmsub_cp_v4r8(val, b.val, c.val) );
	}
};

struct v4r8_1133{
	typedef __builtin_v4r8 _v4r8;
	_v4r8 val;
	v4r8_1133(const v4r8 src) : val(src.val) {}
	v4r8 operator-(const v4r8 rhs) const {
		const v4r8 one(1.0);
		return v4r8( _fjsp_msub_cp_sr1_v4r8(val, one.val, rhs.val) );
	}
	friend v4r8 operator-(const v4r8 lhs, const v4r8_1133 rhs){
		const v4r8 one(1.0);
		return v4r8( _fjsp_nmsub_cp_sr1_v4r8(rhs.val, one.val, lhs.val) );
	}
	v4r8 operator*(const v4r8 rhs) const {
		const _v4r8 zero = _fjsp_setzero_v4r8();
		return v4r8( _fjsp_madd_cp_sr1_v4r8(val, rhs.val, zero) );
	}
	friend v4r8 operator*(const v4r8 lhs, const v4r8_1133 rhs){
		const _v4r8 zero = _fjsp_setzero_v4r8();
		return v4r8( _fjsp_madd_cp_sr1_v4r8(rhs.val, lhs.val, zero) );
	}
	v4r8 madd(const v4r8 b, const v4r8 c) const {
		return v4r8( _fjsp_madd_cp_sr1_v4r8(val, b.val, c.val) );
	}
	v4r8 nmsub(const v4r8 b, const v4r8 c) const {
		return v4r8( _fjsp_nmsub_cp_sr1_v4r8(val, b.val, c.val) );
	}
};
#endif
