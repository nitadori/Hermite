#include <cstdio>
#include "vector3.h"
#define CUDA_TITAN
#include "hermite8.h"
// #include "hermite8-titan.h"

__device__ __forceinline__ void predict_one(
		const double             tsys,
		const Gravity::GParticle &p,
		Gravity::GPredictor      &pr)
{
		const double dt  = tsys - p.tlast;
		const double dt2 = (1./2.) * dt;
		const double dt3 = (1./3.) * dt;
		const double dt4 = (1./4.) * dt;
		const double dt5 = (1./5.) * dt;
		const double dt6 = (1./6.) * dt;
		const double dt7 = (1./7.) * dt;

		double3 pos, vel, acc, jrk;

		pos.x = 
			p.pos.x + dt *(
			p.vel.x + dt2*(
			p.acc.x + dt3*(
			p.jrk.x + dt4*(
			p.snp.x + dt5*(
			p.crk.x + dt6*(
			p.d4a.x + dt7*(
			p.d5a.x )))))));
		pos.y = 
			p.pos.y + dt *(
			p.vel.y + dt2*(
			p.acc.y + dt3*(
			p.jrk.y + dt4*(
			p.snp.y + dt5*(
			p.crk.y + dt6*(
			p.d4a.y + dt7*(
			p.d5a.y )))))));
		pos.z = 
			p.pos.z + dt *(
			p.vel.z + dt2*(
			p.acc.z + dt3*(
			p.jrk.z + dt4*(
			p.snp.z + dt5*(
			p.crk.z + dt6*(
			p.d4a.z + dt7*(
			p.d5a.z )))))));

		vel.x = 
			p.vel.x + dt *(
			p.acc.x + dt2*(
			p.jrk.x + dt3*(
			p.snp.x + dt4*(
			p.crk.x + dt5*(
			p.d4a.x + dt6*(
			p.d5a.x ))))));
		vel.y = 
			p.vel.y + dt *(
			p.acc.y + dt2*(
			p.jrk.y + dt3*(
			p.snp.y + dt4*(
			p.crk.y + dt5*(
			p.d4a.y + dt6*(
			p.d5a.y ))))));
		vel.z = 
			p.vel.z + dt *(
			p.acc.z + dt2*(
			p.jrk.z + dt3*(
			p.snp.z + dt4*(
			p.crk.z + dt5*(
			p.d4a.z + dt6*(
			p.d5a.z ))))));

		acc.x = 
			p.acc.x + dt *(
			p.jrk.x + dt2*(
			p.snp.x + dt3*(
			p.crk.x + dt4*(
			p.d4a.x + dt5*(
			p.d5a.x )))));
		acc.y = 
			p.acc.y + dt *(
			p.jrk.y + dt2*(
			p.snp.y + dt3*(
			p.crk.y + dt4*(
			p.d4a.y + dt5*(
			p.d5a.y )))));
		acc.z = 
			p.acc.z + dt *(
			p.jrk.z + dt2*(
			p.snp.z + dt3*(
			p.crk.z + dt4*(
			p.d4a.z + dt5*(
			p.d5a.z )))));

		jrk.x = 
			p.jrk.x + dt *(
			p.snp.x + dt2*(
			p.crk.x + dt3*(
			p.d4a.x + dt4*(
			p.d5a.x ))));
		jrk.y = 
			p.jrk.y + dt *(
			p.snp.y + dt2*(
			p.crk.y + dt3*(
			p.d4a.y + dt4*(
			p.d5a.y ))));
		jrk.z = 
			p.jrk.z + dt *(
			p.snp.z + dt2*(
			p.crk.z + dt3*(
			p.d4a.z + dt4*(
			p.d5a.z ))));

		pr.pos  = pos;
		pr.mass = p.mass;
		pr.vel  = vel;
		pr.acc  = acc;
		pr.jrk  = jrk;
}

#if 1 // naive version
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

	{
		const double2 *src = (const double2 *)(ptcl+off);
		double2 *dst = (double2 *)(pshare);
		// copy 832 DP words
#pragma unrll
		for(int i=0; i<13; i++){
			dst[32*i + tid] = src[32*i + tid];
		}
	}
	Gravity::GPredictor pr;
	predict_one(tsys, pshare[tid], pr);
	prbuf[tid] = pr;
	{
		const double *src = (const double *)(prbuf);
		double *dst = (double *)(pred + off);
		// copy 416 DP words
#pragma unrll
		for(int i=0; i<13; i++){
			dst[32*i + tid] = src[32*i + tid];
		}
	}
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

