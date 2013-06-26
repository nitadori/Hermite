#pragma once
#include <cstdlib>
#include <cassert>
#include <cstdio>
template <typename T, size_t align>
T *allocate(size_t num_elem){
	void *ptr;
#ifdef __MIC__
	ptr = _mm_malloc(num_elem * sizeof(T), align);
#else
	int ret = posix_memalign(&ptr, align, num_elem * sizeof(T));
	assert(0 == ret);
#endif
	assert(ptr);
	return (T *)ptr;
}

template <typename T, size_t align>
void deallocate(T *ptr){
#ifdef __MIC__
	_mm_free(ptr);
#else
	free(ptr);
#endif
}
