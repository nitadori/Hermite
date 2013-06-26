#pragma once
#include <cassert>
template <int N>
double pow_one_nth_quant(const double x){
	assert(x > 0.0);
	union{
		double d;
		unsigned long l;
	}m64;
	m64.d = x;
	const int dec = 1023 % N;
	const int inc = 1023 - 1023 / N;
	m64.l >>= 52;
	m64.l  -= dec;
	m64.l  /= N;
	m64.l  += inc;
	m64.l <<= 52;
	return m64.d;
}
