#include "allocate.h"
#include "vector3.h"
#include "hermite4.h"
#include "hermite4-k.h"


#if 0
void Gravity::predict_all_rp(
		const int nbody, 
		const double s_tsys, 
		const GParticle * __restrict ptcl,
		GPredictor      * __restrict pred)
{
	const v2r8 tsys(s_tsys);
	const int nb2 = nbody / 2;
#pragma loop noswp
#pragma loop unroll 8
#pragma omp parallel for
	for(int i=0; i<nb2; i++){
		const v2r8 dt  = tsys - ptcl[i].tlast;
		const v2r8 dt2 = v2r8(1./2.) * dt;
		const v2r8 dt3 = v2r8(1./3.) * dt;
#pragma loop unroll
		for(int k=0; k<3; k++){
			pred[i].pos[k] = 
				ptcl[i].pos[k] + dt * (
					ptcl[i].vel[k] + dt2 * (
						ptcl[i].acc[k] + dt3 * (
							ptcl[i].jrk[k])));
			pred[i].vel[k] = 
				ptcl[i].vel[k] + dt * (
					ptcl[i].acc[k] + dt2 * (
						ptcl[i].jrk[k]));

		}
		pred[i].mass = ptcl[i].mass;
	}
}
#endif
void Gravity::predict_all_rp_fast_omp(
		const int nbody, 
		const double s_tsys, 
		const GParticle * __restrict ptcl,
		GPredictor      * __restrict pred)
{
	const v2r8 tsys(s_tsys);
	const int nb2 = nbody / 2;
#pragma loop noswp
#pragma loop unroll 8
#pragma omp for nowait
	for(int i=0; i<nb2; i++){
		const v2r8 dt  = tsys - ptcl[i].tlast;
		const v2r8 dt2 = v2r8(1./2.) * dt;
		const v2r8 dt3 = v2r8(1./3.) * dt;
#pragma loop unroll
		for(int k=0; k<3; k++){
			pred[i].pos[k] = 
				ptcl[i].pos[k] + dt * (
					ptcl[i].vel[k] + dt2 * (
						ptcl[i].acc[k] + dt3 * (
							ptcl[i].jrk[k])));
			pred[i].vel[k] = 
				ptcl[i].vel[k] + dt * (
					ptcl[i].acc[k] + dt2 * (
						ptcl[i].jrk[k]));

		}
		pred[i].mass = ptcl[i].mass;
	}
}

#if 1
#if 0
void Gravity::calc_force_in_range(
		const int    is,
		const int    ie,
		const double eps2_s,
		Force * __restrict force )
{
	static GForce fobuf[MAXTHREAD][NIMAX/2];
	int nthreads;
#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
#pragma omp master
		nthreads = omp_get_num_threads();

		const int nj = nbody;
		for(int i=is, ii=0; i<ie; i+=2, ii++){
			v4r8 ax(0.0);
			v4r8 ay(0.0);
			v4r8 az(0.0);
			v4r8 jx(0.0);
			v4r8 jy(0.0);
			v4r8 jz(0.0);
			const v2r8 xi  = pred[i/2].pos[0];
			const v2r8 yi  = pred[i/2].pos[1];
			const v2r8 zi  = pred[i/2].pos[2];
			const v2r8 vxi = pred[i/2].vel[0];
			const v2r8 vyi = pred[i/2].vel[1];
			const v2r8 vzi = pred[i/2].vel[2];
			const v2r8 eps2(eps2_s);
#pragma omp for nowait // calculate partial force
			for(int j=0; j<nj; j+=2){
				const v2r8 mj  = pred[j/2].mass;
				const v2r8 xj  = pred[j/2].pos[0];
				const v2r8 yj  = pred[j/2].pos[1];
				const v2r8 zj  = pred[j/2].pos[2];
				const v2r8 vxj = pred[j/2].vel[0];
				const v2r8 vyj = pred[j/2].vel[1];
				const v2r8 vzj = pred[j/2].vel[2];

				const v4r8 dx = v4r8_llhh(xj) - v4r8(xi);
				const v4r8 dy = v4r8_llhh(yj) - v4r8(yi);
				const v4r8 dz = v4r8_llhh(zj) - v4r8(zi);
				const v4r8 dvx = v4r8_llhh(vxj) - v4r8(vxi);
				const v4r8 dvy = v4r8_llhh(vyj) - v4r8(vyi);
				const v4r8 dvz = v4r8_llhh(vzj) - v4r8(vzi);

				const v4r8 r2    = ((v4r8(eps2) +dx*dx) + dy*dy) + dz*dz;
				const v4r8 rv = (dx *dvx + dy *dvy) + dz *dvz;

				const v4r8 rinv   = r2.rsqrta_x8();
				const v4r8 rinv2  = rinv * rinv;
				const v4r8 mrinv  = rinv * v4r8_llhh(mj);
				const v4r8 mrinv3 = mrinv * rinv2;

				const v4r8 alpha = v4r8(-3.0) * rv * rinv2;
				
				ax += mrinv3 * dx;
				ay += mrinv3 * dy;
				az += mrinv3 * dz;

				jx += mrinv3 * (dvx + alpha * dx);
				jy += mrinv3 * (dvy + alpha * dy);
				jz += mrinv3 * (dvz + alpha * dz);
			} // for (j)
			fobuf[tid][ii].save(
					ax.hadd(), ay.hadd(), az.hadd(), 
					jx.hadd(), jy.hadd(), jz.hadd());
		} // for(i)
	} // end omp paralle
	// reduction & store
#pragma omp parallel for
	for(int i=is; i<ie; i+=2){
		int ii = (i - is)/2;
		v2r8 ax, ay, az, jx, jy, jz;
#pragma loop noswp
#pragma loop unroll 8
		for(int ith=0; ith<nthreads; ith++){
			// fsum.accumulate(fobuf[ith][ii]);
			ax += fobuf[ith][ii].ax;
			ay += fobuf[ith][ii].ay;
			az += fobuf[ith][ii].az;
			jx += fobuf[ith][ii].jx;
			jy += fobuf[ith][ii].jy;
			jz += fobuf[ith][ii].jz;
		}
		ax.storel(&force[i+0].acc.x);
		ay.storel(&force[i+0].acc.y);
		az.storel(&force[i+0].acc.z);
		jx.storel(&force[i+0].jrk.x);
		jy.storel(&force[i+0].jrk.y);
		jz.storel(&force[i+0].jrk.z);

		ax.storeh(&force[i+1].acc.x);
		ay.storeh(&force[i+1].acc.y);
		az.storeh(&force[i+1].acc.z);
		jx.storeh(&force[i+1].jrk.x);
		jy.storeh(&force[i+1].jrk.y);
		jz.storeh(&force[i+1].jrk.z);
	}
}
#endif

void Gravity::calc_force_in_range_fast_omp(
		const int    is,
		const int    ie,
		const double eps2_s,
		Force * __restrict force )
{
	// static GForce fobuf[MAXTHREAD][NIMAX/2];
	static double buffer[MAXTHREAD*(NIMAX/2)*(sizeof(GForce)/sizeof(double))];
	GForce (*fobuf)[NIMAX/2] = (GForce (*)[NIMAX/2])buffer;

	const int tid = omp_get_thread_num();
	const int nthreads = omp_get_num_threads();

	const int nj = nbody;
	for(int i=is, ii=0; i<ie; i+=2, ii++){
		v4r8 ax(0.0);
		v4r8 ay(0.0);
		v4r8 az(0.0);
		v4r8 jx(0.0);
		v4r8 jy(0.0);
		v4r8 jz(0.0);
		const v2r8 xi  = pred[i/2].pos[0];
		const v2r8 yi  = pred[i/2].pos[1];
		const v2r8 zi  = pred[i/2].pos[2];
		const v2r8 vxi = pred[i/2].vel[0];
		const v2r8 vyi = pred[i/2].vel[1];
		const v2r8 vzi = pred[i/2].vel[2];
		const v2r8 eps2(eps2_s);
#pragma omp for nowait // calculate partial force
		for(int j=0; j<nj; j+=2){
			const v2r8 mj  = pred[j/2].mass;
			const v2r8 xj  = pred[j/2].pos[0];
			const v2r8 yj  = pred[j/2].pos[1];
			const v2r8 zj  = pred[j/2].pos[2];
			const v2r8 vxj = pred[j/2].vel[0];
			const v2r8 vyj = pred[j/2].vel[1];
			const v2r8 vzj = pred[j/2].vel[2];

			const v4r8 dx = v4r8_llhh(xj) - v4r8(xi);
			const v4r8 dy = v4r8_llhh(yj) - v4r8(yi);
			const v4r8 dz = v4r8_llhh(zj) - v4r8(zi);
			const v4r8 dvx = v4r8_llhh(vxj) - v4r8(vxi);
			const v4r8 dvy = v4r8_llhh(vyj) - v4r8(vyi);
			const v4r8 dvz = v4r8_llhh(vzj) - v4r8(vzi);

			const v4r8 r2    = ((v4r8(eps2) +dx*dx) + dy*dy) + dz*dz;
			const v4r8 rv = (dx *dvx + dy *dvy) + dz *dvz;

			const v4r8 rinv   = r2.rsqrta_x8();
			const v4r8 rinv2  = rinv * rinv;
			const v4r8 mrinv  = rinv * v4r8_llhh(mj);
			const v4r8 mrinv3 = mrinv * rinv2;

			const v4r8 alpha = v4r8(-3.0) * rv * rinv2;

			ax += mrinv3 * dx;
			ay += mrinv3 * dy;
			az += mrinv3 * dz;

			jx += mrinv3 * (dvx + alpha * dx);
			jy += mrinv3 * (dvy + alpha * dy);
			jz += mrinv3 * (dvz + alpha * dz);
		} // for (j)
		fobuf[tid][ii].save(
				ax.hadd(), ay.hadd(), az.hadd(), 
				jx.hadd(), jy.hadd(), jz.hadd());
	} // for(i)
#pragma omp barrier
	// reduction & store
	if(0 == is && ie-is <= NACT_PARALLEL_THRESH){ // serial execution
#pragma omp master
		for(int i=is; i<ie; i+=2){
			int ii = (i - is)/2;
			v2r8 ax, ay, az, jx, jy, jz;
#pragma loop noswp
#pragma loop unroll 8
			for(int ith=0; ith<nthreads; ith++){
				// fsum.accumulate(fobuf[ith][ii]);
				ax += fobuf[ith][ii].ax;
				ay += fobuf[ith][ii].ay;
				az += fobuf[ith][ii].az;
				jx += fobuf[ith][ii].jx;
				jy += fobuf[ith][ii].jy;
				jz += fobuf[ith][ii].jz;
			}
			ax.storel(&force[i+0].acc.x);
			ay.storel(&force[i+0].acc.y);
			az.storel(&force[i+0].acc.z);
			jx.storel(&force[i+0].jrk.x);
			jy.storel(&force[i+0].jrk.y);
			jz.storel(&force[i+0].jrk.z);

			ax.storeh(&force[i+1].acc.x);
			ay.storeh(&force[i+1].acc.y);
			az.storeh(&force[i+1].acc.z);
			jx.storeh(&force[i+1].jrk.x);
			jy.storeh(&force[i+1].jrk.y);
			jz.storeh(&force[i+1].jrk.z);
		} // no wait, return and goto corrector
	}else{
#pragma omp for
		for(int i=is; i<ie; i+=2){
			int ii = (i - is)/2;
			v2r8 ax, ay, az, jx, jy, jz;
#pragma loop noswp
#pragma loop unroll 8
			for(int ith=0; ith<nthreads; ith++){
				// fsum.accumulate(fobuf[ith][ii]);
				ax += fobuf[ith][ii].ax;
				ay += fobuf[ith][ii].ay;
				az += fobuf[ith][ii].az;
				jx += fobuf[ith][ii].jx;
				jy += fobuf[ith][ii].jy;
				jz += fobuf[ith][ii].jz;
			}
			ax.storel(&force[i+0].acc.x);
			ay.storel(&force[i+0].acc.y);
			az.storel(&force[i+0].acc.z);
			jx.storel(&force[i+0].jrk.x);
			jy.storel(&force[i+0].jrk.y);
			jz.storel(&force[i+0].jrk.z);

			ax.storeh(&force[i+1].acc.x);
			ay.storeh(&force[i+1].acc.y);
			az.storeh(&force[i+1].acc.z);
			jx.storeh(&force[i+1].jrk.x);
			jy.storeh(&force[i+1].jrk.y);
			jz.storeh(&force[i+1].jrk.z);
		} // here comes a barrier
	}
}

void Gravity::calc_potential_rp(
		const int    nbody,
		const double eps2_s,
		const GParticle * __restrict ptcl,
		v4r8            * __restrict xmbuf,
		double          * __restrict potbuf)
{
	// setup cache
#pragma omp parallel for
	for(int ii=0; ii<nbody; ii+=2){
		const v2r8 x2 = ptcl[ii/2].pos[0];
		const v2r8 y2 = ptcl[ii/2].pos[1];
		const v2r8 z2 = ptcl[ii/2].pos[2];
		const v2r8 m2 = ptcl[ii/2].mass;
		const v4r8 xm0(v2r8::unpckl(x2, y2), v2r8::unpckl(z2, m2));
		const v4r8 xm1(v2r8::unpckh(x2, y2), v2r8::unpckh(z2, m2));
		xmbuf[ii + 0] = xm0;
		xmbuf[ii + 1] = xm1;
	}
	// evaluate potential, simple i-parallel
#pragma omp parallel for
	for(int i=0; i<nbody; i+=4){
		double (*xm4)[4] = (double (*)[4])(xmbuf + i);
		const v4r8 xi(xm4[0][0], xm4[1][0], xm4[2][0], xm4[3][0]);
		const v4r8 yi(xm4[0][1], xm4[1][1], xm4[2][1], xm4[3][1]);
		const v4r8 zi(xm4[0][2], xm4[1][2], xm4[2][2], xm4[3][2]);
		const v4r8 eps2(eps2_s);
		v4r8 pot(0.0);
		for(int j=0; j<nbody; j++){
			const v4r8 xmj = xmbuf[j];
			const v4r8 dx = xi - v2r8_bcl(xmj.v0);
			const v4r8 dy = yi - v2r8_bch(xmj.v0);
			const v4r8 dz = zi - v2r8_bcl(xmj.v1);

			const v4r8 r2    = ((v4r8(eps2) +dx*dx) + dy*dy) + dz*dz;
			const v4r8 rinv   = (r2.rsqrta_x8()) & (eps2 < r2);

			// pot -= mj * rinv;
			pot.v0 = __builtin_fj_nmsub_cp_sr1_v2r8(xmj.v1, rinv.v0, pot.v0);
			pot.v1 = __builtin_fj_nmsub_cp_sr1_v2r8(xmj.v1, rinv.v1, pot.v1);
		}
		*(v4r8 *)(potbuf + i) = pot;
	}
}
#endif
