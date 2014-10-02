#include <cstdio>
#include "vector3.h"
#define CUDA_TITAN
#include "hermite4.h"
// #include "hermite4-titan.h"

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

		const double dt = tsys - p.tlast;
		const double dt2 = (1./2.) * dt;;
		const double dt3 = (1./3.) * dt;;

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
}

void Gravity::predict_all(const double tsys){
	ptcl.htod(njpsend);
	
	const int nblock = (nbody/NTHREAD) + 
	                  ((nbody%NTHREAD) ? 1 : 0);
	predict_kernel <<<nblock, NTHREAD>>>
		(nbody, ptcl, pred, tsys);

	pred.dtoh();
	puts("pred all done");
	exit(1);
}

enum{
	NJBLOCK = Gravity::NJBLOCK,
};

__global__ void force_kernel(
		const int                  is,
		const int                  ie,
		const int                  nj,
		const Gravity::GPredictor *pred,
		const double               eps2,
		Gravity::GForce          (*fo)[NJBLOCK])
{
}

__global__ void reduce_kernel(
		const Gravity::GForce (*fpart)[NJBLOCK],
		Gravity::GForce        *ftot)
{
}

void Gravity::calc_force_in_range(
	   	const int    is,
		const int    ie,
		const double eps2,
		Force        force[] )
{
	const int ni = ie - is;
	const int niblock = (ni/NTHREAD) + 
	                   ((ni%NTHREAD) ? 1 : 0);
	dim3 grid(niblock, NJBLOCK, 1);
	force_kernel <<<grid, NTHREAD>>>
		(is, ie, nbody, pred, eps2, fpart);

	reduce_kernel <<<ni, NJREDUCE>>>
		(fpart, ftot);
}

#include "pot-titan.hu"
