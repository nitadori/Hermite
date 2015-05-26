#include "allocate.h"
#include "vector3.h"
#include "hermite4.h"
#include "hermite4-mx.h"

void Gravity::predict_all_rp_fast_omp(
		const int nbody, 
		const double s_tsys, 
		const GParticle * __restrict ptcl,
		GPredictor      * __restrict pred)
{
	v4r8::simd_mode_4();

	const v4r8 tsys(s_tsys);

#pragma omp for nowait
	for(int i=0; i<nbody; i++){
		const v4r8 dt = tsys - v4r8::broadcastload(&ptcl[i].tlast);
		const v4r8 c(1./2., 1./3., 1/2., 1./3.);
		const v4r8 dt2 = dt * v4r8_0022(c);
		const v4r8 dt3 = dt * v4r8_1133(c);

		v4r8 pos  = v4r8::load(ptcl[i].pos);
		v4r8 vel  = v4r8::load(ptcl[i].vel);
		v4r8 acc  = v4r8::load(ptcl[i].acc);
		v4r8 jrk  = v4r8::load(ptcl[i].jrk);
		v4r8 mass = v4r8::load(&ptcl[i].mass);

		pos = pos + dt*(vel + dt2*(acc + dt3*jrk));
		vel = vel + dt*(acc + dt2*(jrk));
		pos = pos.ecsl(3).ecsl(mass, 1);

		pos.store(pred[i].pos);
		vel.store(pred[i].vel);
	}

	// v4r8::simd_mode_2();
}

void Gravity::calc_force_in_range_fast_omp(
		const int    is,
		const int    ie,
		const double eps2_s,
		Force * __restrict force )
{
}

void Gravity::calc_potential_rp(
		const int    nbody,
		const double deps2,
		const GParticle * __restrict ptcl,
		v4r8            * __restrict xmbuf,
		double          * __restrict potbuf)
{
#pragma omp parallel
	{
		v4r8::simd_mode_4();
		asm volatile("ssm 1"); // WTF!!
		//setup cache
#pragma omp for
		for(int i=0; i<nbody; i++){
			xmbuf[i] = v4r8::load(ptcl[i].pos);
		}
		// evaluate potential, simple i-parallel
		v4r8 eps2(deps2);
#pragma omp for
		for(int i=0; i<nbody; i+=4){
			v4r8 xi = xmbuf[i+0];
			v4r8 yi = xmbuf[i+1];
			v4r8 zi = xmbuf[i+2];
			v4r8 mi = xmbuf[i+3];

			v4r8::transpose(xi, yi, zi, mi);

			v4r8 pot(0.0);

			for(int j=0; j<nbody; j++){
				const double *ptr = (const double *)(xmbuf + j);
				v4r8 xj = v4r8::broadcastload(ptr + 0);
				v4r8 yj = v4r8::broadcastload(ptr + 1);
				v4r8 zj = v4r8::broadcastload(ptr + 2);
				v4r8 mj = v4r8::broadcastload(ptr + 3);

				v4r8 dx = xj - xi;
				v4r8 dy = yj - yi;
				v4r8 dz = zj - zi;

				v4r8 r2    = ((eps2 +dx*dx) + dy*dy) + dz*dz;
				v4r8 rinv  = (r2.rsqrta_x8()) & (eps2 < r2);

				pot -= mj * rinv;
			}
			pot.store(potbuf + i);
		}
	}
}
