#include <cstdio>
#include "vector3.h"
#define CUDA_TITAN
#include "hermite8.h"
// #include "hermite8-titan.h"
#include "cuda-common.hu"

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
// 26N DP -> 13N DP
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

	static_memcpy<double2, 32*13, 32> (pshare, ptcl+off);

	Gravity::GPredictor pr;
	predict_one(tsys, pshare[tid], pr);
	prbuf[tid] = pr;

	static_memcpy<double, 32*13, 32> (pred+off, prbuf);
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
		double3                    &jrk,
		double3                    &snp,
		double3                    &crk)
{
		const double dx  = jpred.pos.x - ipred.pos.x;
		const double dy  = jpred.pos.y - ipred.pos.y;
		const double dz  = jpred.pos.z - ipred.pos.z;

		const double dvx = jpred.vel.x - ipred.vel.x;
		const double dvy = jpred.vel.y - ipred.vel.y;
		const double dvz = jpred.vel.z - ipred.vel.z;

		const double dax = jpred.acc.x - ipred.acc.x;
		const double day = jpred.acc.y - ipred.acc.y;
		const double daz = jpred.acc.z - ipred.acc.z;

		const double djx = jpred.jrk.x - ipred.jrk.x;
		const double djy = jpred.jrk.y - ipred.jrk.y;
		const double djz = jpred.jrk.z - ipred.jrk.z;

		const double mj  = jpred.mass;

		const double dr2  = eps2 + dx*dx + dy*dy + dz*dz;
		const double drdv =  dx*dvx +  dy*dvy +  dz*dvz;
		const double dvdv = dvx*dvx + dvy*dvy + dvz*dvz;
		const double drda =  dx*dax +  dy*day +  dz*daz;
		const double dvda = dvx*dax + dvy*day + dvz*daz;
		const double drdj =  dx*djx +  dy*djy +  dz*djz;

		const double rinv1 = rsqrt(dr2);
		const double rinv2 = rinv1 * rinv1;
		const double mrinv3 = mj * rinv1 * rinv2;

		double alpha = drdv * rinv2;
		double beta  = (dvdv + drda) * rinv2 + alpha * alpha;
		double gamma = (3.0*dvda + drdj)*rinv2 + alpha*(3.0*beta - 4.0*alpha*alpha);

		acc.x += mrinv3 * dx;
		acc.y += mrinv3 * dy;
		acc.z += mrinv3 * dz;

		alpha *= -3.0;
		const double  tx = dvx + alpha * dx;
		const double  ty = dvy + alpha * dy;
		const double  tz = dvz + alpha * dz;
		jrk.x += mrinv3 * tx;
		jrk.y += mrinv3 * ty;
		jrk.z += mrinv3 * tz;

		alpha *= 2.0;
		beta *= -3.0;
		const double ux = dax + alpha * tx + beta * dx;
		const double uy = day + alpha * ty + beta * dy;
		const double uz = daz + alpha * tz + beta * dz;
		snp.x += mrinv3 * ux;
		snp.y += mrinv3 * uy;
		snp.z += mrinv3 * uz;

		alpha *=1.5;
		beta  *= 3.0;
		gamma *= -3.0;
		crk.x += mrinv3 * (djx + alpha * ux + beta * tx + gamma * dx);
		crk.y += mrinv3 * (djy + alpha * uy + beta * ty + gamma * dy);
		crk.z += mrinv3 * (djz + alpha * uz + beta * tz + gamma * dz);
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
		double3 snp = make_double3(0.0, 0.0, 0.0);
		double3 crk = make_double3(0.0, 0.0, 0.0);

#pragma unroll 4
		for(int j=js; j<je; j++){
			const Gravity::GPredictor &jpred = pred[j];
			pp_interact(ipred, jpred, eps2, acc, jrk, snp, crk);
			
		}

		fo[xid][yid].acc = acc;
		fo[xid][yid].jrk = jrk;
		fo[xid][yid].snp = snp;
		fo[xid][yid].crk = crk;
	}
}
#else // tuned with shared memory
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

#if 1
	__shared__ Gravity::GPredictor jpsh[8];
#else
	__shared__ double2 align_buf[52];
	Gravity::GPredictor *jpsh = (Gravity::GPredictor *)(align_buf);
#endif

	const Gravity::GPredictor ipred = pred[i];
	double3 acc = make_double3(0.0, 0.0, 0.0);
	double3 jrk = make_double3(0.0, 0.0, 0.0);
	double3 snp = make_double3(0.0, 0.0, 0.0);
	double3 crk = make_double3(0.0, 0.0, 0.0);

	for(int j=js; j<je8; j+=8){
		__syncthreads();
		// static_memcpy<double2, 52, Gravity::NTHREAD> (jpsh, pred + j); // this doesn't run
		static_memcpy<double, 104, Gravity::NTHREAD> (jpsh, pred + j);
		// 104 = sizeof(jpsh) / sizeof(double)
		__syncthreads();

#pragma unroll
		for(int jj=0; jj<8; jj++){
			const Gravity::GPredictor &jpred = jpsh[jj];
			pp_interact(ipred, jpred, eps2, acc, jrk, snp, crk);
		}
	}

	__syncthreads();
	static_memcpy<double, 104, Gravity::NTHREAD> (jpsh, pred + je8);
	__syncthreads();

	for(int j=je8; j<je; j++){
		const Gravity::GPredictor &jpred = jpsh[j - je8];
		pp_interact(ipred, jpred, eps2, acc, jrk, snp, crk);
	}

	if(i < ie){
		fo[xid][yid].acc = acc;
		fo[xid][yid].jrk = jrk;
		fo[xid][yid].snp = snp;
		fo[xid][yid].crk = crk;
	}
}
#endif

__global__ void reduce_kernel(
		const Gravity::GForce (*fpart)[NJBLOCK],
		Gravity::GForce        *ftot)
{
	const int bid = blockIdx.x;  // for particle
	const int xid = threadIdx.x; // for 56 partial force
	const int yid = threadIdx.y; // for 12 elements of Force

	const Gravity::GForce &fsrc = fpart[bid][xid];
	const double          *dsrc = (const double *)(&fsrc);
	
	const double x = xid<NJBLOCK ? dsrc[yid] : 0.0;
	const double y = warp_reduce_double(x);

	Gravity::GForce &fdst = ftot[bid];
	double          *ddst = (double *)(&fdst);
	if(32 == Gravity::NJREDUCE){
		if(0==xid) ddst[yid] = y;
	}
	if(64 == Gravity::NJREDUCE){
		// neeeds inter-warp reduction
		__shared__ double fsh[12][2];
		fsh[yid][xid/32] = y;
		__syncthreads();
		if(0==xid) ddst[yid] = fsh[yid][0] + fsh[yid][1];
	}
}

void Gravity::calc_force_in_range(
	   	const int    is,
		const int    ie,
		const double eps2,
		Force        force[] )
{
	assert(104 == sizeof(GPredictor));
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
		assert(12 == nword);
		reduce_kernel <<<ni, dim3(NJREDUCE, nword, 1)>>>
			(fpart, ftot);
	}

	ftot.dtoh(ni);
	for(int i=0; i<ni; i++){
		force[is+i].acc.x = ftot[i].acc.x;
		force[is+i].acc.y = ftot[i].acc.y;
		force[is+i].acc.z = ftot[i].acc.z;
		force[is+i].jrk.x = ftot[i].jrk.x;
		force[is+i].jrk.y = ftot[i].jrk.y;
		force[is+i].jrk.z = ftot[i].jrk.z;
		force[is+i].snp.x = ftot[i].snp.x;
		force[is+i].snp.y = ftot[i].snp.y;
		force[is+i].snp.z = ftot[i].snp.z;
		force[is+i].crk.x = ftot[i].crk.x;
		force[is+i].crk.y = ftot[i].crk.y;
		force[is+i].crk.z = ftot[i].crk.z;
	}
}

#include "pot-titan.hu"

