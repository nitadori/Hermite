#ifdef __SSE__
#warning SSE is available
typedef double v2df __attribute__((vector_size(16)));
#endif
#ifdef __AVX__
#warning AVX is available
typedef double v4df __attribute__((vector_size(32)));
#endif
#ifdef __AVX2__
#warning AVX2 is available
#endif

#include <cstdio>
#include "vector3.h"
#define CUDA_TITAN
#include "hermite4.h"
// #include "hermite4-titan.h"
#include "cuda-common.hu"

__device__ __forceinline__ void predict_one(
		const double             tsys,
		const Gravity::GParticle &p,
		Gravity::GPredictor      &pr)
{
		const double dt  = tsys - p.tlast;
		const double dt2 = (1./2.) * dt;
		const double dt3 = (1./3.) * dt;

		double3 pos, vel;
		pos.x = 
			p.pos.x + dt *(
			p.vel.x + dt2*(
			p.acc.x + dt3*(
			p.jrk.x )));
		pos.y = 
			p.pos.y + dt *(
			p.vel.y + dt2*(
			p.acc.y + dt3*(
			p.jrk.y )));
		pos.z = 
			p.pos.z + dt *(
			p.vel.z + dt2*(
			p.acc.z + dt3*(
			p.jrk.z )));
		vel.x = 
			p.vel.x + dt *(
			p.acc.x + dt2*(
			p.jrk.x ));
		vel.y = 
			p.vel.y + dt *(
			p.acc.y + dt2*(
			p.jrk.y ));
		vel.z = 
			p.vel.z + dt *(
			p.acc.z + dt2*(
			p.jrk.z ));

		pr.pos  = pos;
		pr.mass = p.mass;
		pr.vel  = vel;
}

#if 0 // naive version
__global__ void predict_kernel(
		const int                 nbody,
		const Gravity::GParticle *ptcl,
		Gravity::GPredictor      *pred,
		const double              tsys)
{
	const int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < nbody){
		Gravity::GParticle   p  = ptcl[tid];
		Gravity::GPredictor &pr = pred[tid];
		predict_one(tsys, p, pr);

	}
}
#else // specialized for 32 threads
// 14N DP -> 7N DP
__global__ void predict_kernel(
		const int                 nbody,
		const Gravity::GParticle *ptcl,
		Gravity::GPredictor      *pred,
		const double              tsys)
{
	const int tid = threadIdx.x;
	const int off = blockDim.x * blockIdx.x;

	__shared__ Gravity::GParticle pshare[32];
	Gravity::GPredictor *prbuf = (Gravity::GPredictor *)pshare;

	static_memcpy<double2, 32*7, 32> (pshare, ptcl+off);

	Gravity::GPredictor pr;
	predict_one(tsys, pshare[tid], pr);
	prbuf[tid] = pr;

	static_memcpy<double, 32*7, 32> (pred+off, prbuf);
}
#endif

void Gravity::predict_all(const double tsys){
	ptcl.htod(njpsend);
	// printf("sent %d stars\n", njpsend);

	const int ntpred = 32;
	
	const int nblock = (nbody/ntpred) + 
	                  ((nbody%ntpred) ? 1 : 0);
	predict_kernel <<<nblock, ntpred>>>
		(nbody, ptcl, pred, tsys);

	// pred.dtoh(); // THIS DEBUGGING LINE WAS THE BOTTLENECK
	// puts("pred all done");
	cudaThreadSynchronize(); // for profiling
}

enum{
	NJBLOCK = Gravity::NJBLOCK,
};

__device__ __forceinline__ void pp_interact(
		const Gravity::GPredictor &ipred,
		const Gravity::GPredictor &jpred,
		const double                eps2,
		double3                    &acc,
		double3                    &jrk)
{
		const double dx  = jpred.pos.x - ipred.pos.x;
		const double dy  = jpred.pos.y - ipred.pos.y;
		const double dz  = jpred.pos.z - ipred.pos.z;
		const double dvx = jpred.vel.x - ipred.vel.x;
		const double dvy = jpred.vel.y - ipred.vel.y;
		const double dvz = jpred.vel.z - ipred.vel.z;
		const double mj  = jpred.mass;

		const double dr2  = eps2 + dx*dx + dy*dy + dz*dz;
		const double drdv = dx*dvx + dy*dvy + dz*dvz;

		// const double rinv1 = rsqrt(dr2);
		const double rinv1 = rsqrt_x3(dr2);
		const double rinv2 = rinv1 * rinv1;
		const double mrinv3 = mj * rinv1 * rinv2;

		double alpha = drdv * rinv2;
		alpha *= -3.0;

		acc.x += mrinv3 * dx;
		acc.y += mrinv3 * dy;
		acc.z += mrinv3 * dz;
		jrk.x += mrinv3 * (dvx + alpha * dx);
		jrk.y += mrinv3 * (dvy + alpha * dy);
		jrk.z += mrinv3 * (dvz + alpha * dz);
}

#if 0  // first version
__global__ void force_kernel(
		const int                  is,
		const int                  ie,
		const int                  nj,
		const Gravity::GPredictor *pred,
		const double               eps2,
		Gravity::GForce          (*fo)[NJBLOCK])
{
	const int xid = threadIdx.x + blockDim.x * blockIdx.x;
	const int yid = blockIdx.y;

	const int js = ((0 + yid) * nj) / NJBLOCK;
	const int je = ((1 + yid) * nj) / NJBLOCK;

	const int i = is + xid;
	if(i < ie){
		const Gravity::GPredictor ipred = pred[i];
		double3 acc = make_double3(0.0, 0.0, 0.0);
		double3 jrk = make_double3(0.0, 0.0, 0.0);

#pragma unroll 4
		for(int j=js; j<je; j++){
			const Gravity::GPredictor &jpred = pred[j];
			pp_interact(ipred, jpred, eps2, acc, jrk);
			
		}

		fo[xid][yid].acc = acc;
		fo[xid][yid].jrk = jrk;
	}
}
#else
__global__ void force_kernel(
		const int                  is,
		const int                  ie,
		const int                  nj,
		const Gravity::GPredictor *pred,
		const double               eps2,
		Gravity::GForce          (*fo)[NJBLOCK])
{
	// const int tid = threadIdx.x;
	const int xid = threadIdx.x + blockDim.x * blockIdx.x;
	const int yid = blockIdx.y;

	const int js = ((0 + yid) * nj) / NJBLOCK;
	const int je = ((1 + yid) * nj) / NJBLOCK;
	const int je8 = js + 8*((je-js)/8);

	const int i = is + xid;

	__shared__ Gravity::GPredictor jpsh[8];

	const Gravity::GPredictor ipred = pred[i];
	double3 acc = make_double3(0.0, 0.0, 0.0);
	double3 jrk = make_double3(0.0, 0.0, 0.0);

	for(int j=js; j<je8; j+=8){
		__syncthreads();
		static_memcpy<double, 56, Gravity::NTHREAD> (jpsh, pred + j);
		// 56 = sizeof(jpsh)/sizeof(double)
		__syncthreads();
#pragma unroll
		for(int jj=0; jj<8; jj++){
			// const Gravity::GPredictor &jpred = pred[j+jj];
			const Gravity::GPredictor &jpred = jpsh[jj];
			pp_interact(ipred, jpred, eps2, acc, jrk);
		}
	}
	__syncthreads();
	static_memcpy<double, 56, Gravity::NTHREAD> (jpsh, pred + je8);
	__syncthreads();
	for(int j=je8; j<je; j++){
		// const Gravity::GPredictor &jpred = pred[j];
		const Gravity::GPredictor &jpred = jpsh[j - je8];
		pp_interact(ipred, jpred, eps2, acc, jrk);
	}

	if(i < ie){
		fo[xid][yid].acc = acc;
		fo[xid][yid].jrk = jrk;
	}
}
#endif

#if 0 // was slower
enum{
	NXTH = 32,
	NYTH =  4,
};

__global__ void force_kernel_warp(
		const int                  is,
		const int                  ie,
		const int                  nj,
		const Gravity::GPredictor *pred,
		const double               eps2,
		Gravity::GForce          (*fo)[NJBLOCK])
{
	const int tid = threadIdx.x;
	const int uid = threadIdx.y;
	const int xid = threadIdx.x + blockDim.x * blockIdx.x;
	const int yid = threadIdx.y + blockDim.y * blockIdx.y;

	const int js = ((0 + yid) * nj) / (NJBLOCK*NYTH);
	const int je = ((1 + yid) * nj) / (NJBLOCK*NYTH);
	const int je8 = js + 8*((je-js)/8);

	const int i = is + xid;

	__shared__ Gravity::GPredictor jpsh[NYTH][8];

	const Gravity::GPredictor ipred = pred[i];
	double3 acc = make_double3(0.0, 0.0, 0.0);
	double3 jrk = make_double3(0.0, 0.0, 0.0);

	for(int j=js; j<je8; j+=8){
		static_memcpy<double, 56, Gravity::NTHREAD> (jpsh[uid], pred + j);
		// 56 = sizeof(jpsh)/sizeof(double)
#pragma unroll
		for(int jj=0; jj<8; jj++){
			// const Gravity::GPredictor &jpred = pred[j+jj];
			const Gravity::GPredictor &jpred = jpsh[uid][jj];
			pp_interact(ipred, jpred, eps2, acc, jrk);
		}
	}
	static_memcpy<double, 56, Gravity::NTHREAD> (jpsh[uid], pred + je8);
	for(int j=je8; j<je; j++){
		// const Gravity::GPredictor &jpred = pred[j];
		const Gravity::GPredictor &jpred = jpsh[uid][j - je8];
		pp_interact(ipred, jpred, eps2, acc, jrk);
	}

	acc.x = vreduce<NXTH, NYTH> (acc.x, jpsh);
	acc.y = vreduce<NXTH, NYTH> (acc.y, jpsh);
	acc.z = vreduce<NXTH, NYTH> (acc.z, jpsh);
	jrk.x = vreduce<NXTH, NYTH> (jrk.x, jpsh);
	jrk.y = vreduce<NXTH, NYTH> (jrk.y, jpsh);
	jrk.z = vreduce<NXTH, NYTH> (jrk.z, jpsh);

	if(i < ie && 0==uid){
		fo[xid][yid].acc = acc;
		fo[xid][yid].jrk = jrk;
	}
}
#endif

template<>
__device__ void reduce_final<1, 6>(const double x, double *dst){
	const int yid = threadIdx.y;
	dst[yid] = x;
}

__global__ void reduce_kernel(
		const Gravity::GForce (*fpart)[NJBLOCK],
		Gravity::GForce        *ftot)
{
	const int bid = blockIdx.x;  // for particle
	const int xid = threadIdx.x; // for 56 partial force
	const int yid = threadIdx.y; // for 6 elements of Force

	const Gravity::GForce &fsrc = fpart[bid][xid];
	const double          *dsrc = (const double *)(&fsrc);
	
	const double x = xid<NJBLOCK ? dsrc[yid] : 0.0;
	const double y = warp_reduce_double(x);

	Gravity::GForce &fdst = ftot[bid];
	double          *ddst = (double *)(&fdst);

	reduce_final<Gravity::NJREDUCE/32, 6> (y, ddst);
}

void Gravity::calc_force_in_range(
	   	const int    is,
		const int    ie,
		const double eps2,
		Force        force[] )
{
	assert(56 == sizeof(GPredictor));
	const int ni = ie - is;
	{
		const int niblock = (ni/NTHREAD) + 
						   ((ni%NTHREAD) ? 1 : 0);
		dim3 grid(niblock, NJBLOCK, 1);
		force_kernel <<<grid, NTHREAD>>>
			(is, ie, nbody, pred, eps2, fpart);
	}

	{
		// const int nwarp = 32;
		const int nword = sizeof(GForce) / sizeof(double);
		assert(6 == nword);
		reduce_kernel <<<ni, dim3(NJREDUCE, nword, 1)>>>
			(fpart, ftot);
	}

	ftot.dtoh(ni);
	for(int i=0; i<ni; i++){
		GForce f = ftot[i];
		force[is+i].acc = f.acc;
		force[is+i].jrk = f.jrk;
	}
}

// optimization for overlapping
void Gravity::calc_force_on_first_nact(
		const int    nact,
		const double eps2,
		Force        force[] )
{
	int istore  = 0;
	int nistore = 0;
	for(int ii=0; ii<nact; ii+=NIMAX){
		const int ni = (nact-ii) < NIMAX ? (nact-ii) : NIMAX;
		// calc_force_in_range(ii, ii+ni, eps2, force);
		{   // partial force calcculation
			const int is = ii;
			const int ie = is + ni;
#if 1
			const int niblock = (ni/NTHREAD) + 
				((ni%NTHREAD) ? 1 : 0);
			dim3 grid(niblock, NJBLOCK, 1);
			force_kernel <<<grid, NTHREAD>>>
				(is, ie, nbody, pred, eps2, fpart);
#else
			const int niblock = (ni/32) + 
			                   ((ni%32) ? 1 : 0);
			dim3 grid(niblock, NJBLOCK, 1);
			dim3 thread(NXTH, NYTH, 1);
			force_kernel_warp <<<grid, thread>>>
				(is, ie, nbody, pred, eps2, fpart);
#endif
		}
		for(int i=0; i<nistore; i++){
			GForce f = ftot[i];
			force[istore+i].acc = f.acc;
			force[istore+i].jrk = f.jrk;
		}
		{   // reduction
			const int nword = sizeof(GForce) / sizeof(double);
			assert(6 == nword);
			reduce_kernel <<<ni, dim3(NJREDUCE, nword, 1)>>>
				(fpart, ftot);
		}
		ftot.dtoh(ni);
		istore  = ii;
		nistore = ni;
	}
	for(int i=0; i<nistore; i++){
		GForce f = ftot[i];
		force[istore+i].acc = f.acc;
		force[istore+i].jrk = f.jrk;
	}

	this->njpsend = nact;
}

#include "pot-titan.hu"
