#pragma once
#include "v2r8.h"

struct v4r8_mask{
	typedef __builtin_v2r8 _v2r8;
	_v2r8 v0, v1;

	v4r8_mask(const v4r8_mask &rhs) : v0(rhs.v0), v1(rhs.v1) {}
	v4r8_mask operator=(const v4r8_mask rhs){
		v0 = rhs.v0;
		v1 = rhs.v1;
		return (*this);
	}
	v4r8_mask(const _v2r8 w1, const _v2r8 w2) : v0(w1), v1(w2) {}
	v4r8_mask(const v2r8_mask w1, const v2r8_mask w2) : v0(w1.val), v1(w2.val) {}
};

struct v4r8{
	typedef __builtin_v2r8 _v2r8;
	_v2r8 v0, v1;

	v4r8(const v4r8 &rhs) : v0(rhs.v0), v1(rhs.v1) {}
	v4r8 operator=(const v4r8 rhs){
		v0 = rhs.v0;
		v1 = rhs.v1;
		return (*this);
	}

	v4r8() : 
		v0(__builtin_fj_setzero_v2r8()),
		v1(__builtin_fj_setzero_v2r8())  {}
	v4r8(const double a) : 
		v0(__builtin_fj_set_v2r8(a, a)),
		v1(__builtin_fj_set_v2r8(a, a)) {}
	v4r8(const double a, const double b, const double c, const double d) : 
		v0(__builtin_fj_set_v2r8(a, b)),
		v1(__builtin_fj_set_v2r8(c, d)) {}
	v4r8(const _v2r8 w) : v0(w), v1(w) {}
	v4r8(const _v2r8 w1, const _v2r8 w2) : v0(w1), v1(w2) {}
	v4r8(const v2r8 w) : v0(w.val), v1(w.val) {}
	v4r8(const v2r8 w1, const v2r8 w2) : v0(w1.val), v1(w2.val) {}

	v4r8 operator+(const v4r8 rhs) const {
		return v4r8(
				__builtin_fj_add_v2r8(v0, rhs.v0),
				__builtin_fj_add_v2r8(v1, rhs.v1));
	}
	v4r8 operator-(const v4r8 rhs) const {
		return v4r8(
				__builtin_fj_sub_v2r8(v0, rhs.v0),
				__builtin_fj_sub_v2r8(v1, rhs.v1));
	}
	v4r8 operator*(const v4r8 rhs) const {
		return v4r8(
				__builtin_fj_mul_v2r8(v0, rhs.v0),
				__builtin_fj_mul_v2r8(v1, rhs.v1));
	}

	v4r8 operator+=(const v4r8 rhs){
		return ( (*this) = (*this) + rhs );
	}
	v4r8 operator-=(const v4r8 rhs){
		return ( (*this) = (*this) - rhs );
	}
	v4r8 operator*=(const v4r8 rhs){
		return ( (*this) = (*this) * rhs );
	}
	__attribute__((always_inline))
	v4r8 rsqrta() const {
		return v4r8(
				v2r8(v0).rsqrta(),
				v2r8(v1).rsqrta());
	}
	__attribute__((always_inline))
	v4r8 rsqrta_x8() const {
		return v4r8(
				v2r8(v0).rsqrta_x8(),
				v2r8(v1).rsqrta_x8());
	}
	v4r8 rsqrta_x7() const {
		return v4r8(
				v2r8(v0).rsqrta_x7(),
				v2r8(v1).rsqrta_x7());
	}

	v2r8 first() const {
		return v2r8(v0);
	}
	v2r8 second() const {
		return v2r8(v1);
	}

	v2r8 hadd() const {
		// return v2r8(v0) + v2r8(v1);
		return v2r8(__builtin_fj_add_v2r8(v0, v1));
	}
	void print(
			FILE *fp = stdout,
			const char *fmt = "a = %f, b = %f, c = %f, d = %f\n") const
	{
		double a, b, c, d;
		first ().storel(&a);
		first ().storeh(&b);
		second().storel(&c);
		second().storeh(&d);
		fprintf(fp, fmt, a, b, c, d);
	}

	friend v4r8 operator-(const v4r8 lhs, const v2r8_bcl rhs){
		return v4r8(lhs.first() - rhs, lhs.second() - rhs);
	}
	friend v4r8 operator-(const v4r8 lhs, const v2r8_bch rhs){
		return v4r8(lhs.first() - rhs, lhs.second() - rhs);
	}

	v4r8_mask operator<(const v4r8 rhs) const {
		return v4r8_mask(
				__builtin_fj_cmplt_v2r8(v0, rhs.v0),
				__builtin_fj_cmplt_v2r8(v1, rhs.v1));
	}
	v4r8 operator&(const v4r8_mask rhs){
		return v4r8(
				__builtin_fj_and_v2r8(v0, rhs.v0),
				__builtin_fj_and_v2r8(v1, rhs.v1));
	}
};

struct v4r8_llhh{
	typedef __builtin_v2r8 _v2r8;
	_v2r8 val;

	v4r8_llhh(const v4r8_llhh &rhs) : val(rhs.val) {}
	v4r8_llhh operator=(const v4r8_llhh rhs){
		val = rhs.val;
		return (*this);
	}

	v4r8_llhh(const v2r8 src) : val(src.val) {}

	v4r8 operator+(const v4r8 rhs) const {
		return v4r8(
				v2r8_bcl(val) + rhs.first(),
				v2r8_bch(val) + rhs.second());
	}
	v4r8 operator-(const v4r8 rhs) const {
		return v4r8(
				v2r8_bcl(val) - rhs.first(),
				v2r8_bch(val) - rhs.second());
	}
	v4r8 operator*(const v4r8 rhs) const {
		return v4r8(
				v2r8_bcl(val) * rhs.first(),
				v2r8_bch(val) * rhs.second());
	}

	friend v4r8 operator+(const v4r8 lhs, const v4r8_llhh rhs){
		return v4r8(
				lhs.first()  + v2r8_bcl(rhs.val),
				lhs.second() + v2r8_bch(rhs.val));
	}
	friend v4r8 operator-(const v4r8 lhs, const v4r8_llhh rhs){
		return v4r8(
				lhs.first()  - v2r8_bcl(rhs.val),
				lhs.second() - v2r8_bch(rhs.val));
	}
	friend v4r8 operator*(const v4r8 lhs, const v4r8_llhh rhs){
		return v4r8(
				lhs.first()  * v2r8_bcl(rhs.val),
				lhs.second() * v2r8_bch(rhs.val));
	}
};
