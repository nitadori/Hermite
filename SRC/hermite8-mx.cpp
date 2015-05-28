#include "allocate.h"
#include "vector3.h"
#include "hermite8.h"
#include "hermite8-mx.h"

void Gravity::predict_all_rp_fast_omp(
		const int nbody, 
		const double s_tsys, 
		const GParticle * __restrict ptcl,
		GPredictor      * __restrict pred)
{
	v4r8::simd_mode_4();

	const v4r8 tsys(s_tsys);

#pragma loop noswp
#pragma loop unroll 8
#pragma omp for nowait
	for(int i=0; i<nbody; i++){
		const v4r8 dt = tsys - v4r8::broadcastload(&ptcl[i].tlast);
		const v4r8 c23(1./2., 1./3., 1./2., 1./3.);
		const v4r8 c45(1./4., 1./5., 1./4., 1./5.);
		const v4r8 c67(1./6., 1./7., 1./6., 1./7.);

		const v4r8 dt2 = dt * v4r8_0022(c23);
		const v4r8 dt3 = dt * v4r8_1133(c23);
		const v4r8 dt4 = dt * v4r8_0022(c45);
		const v4r8 dt5 = dt * v4r8_1133(c45);
		const v4r8 dt6 = dt * v4r8_0022(c67);
		const v4r8 dt7 = dt * v4r8_1133(c67);

		v4r8 pos  = v4r8::load(ptcl[i].pos);
		v4r8 vel  = v4r8::load(ptcl[i].vel);
		v4r8 acc  = v4r8::load(ptcl[i].acc);
		v4r8 jrk  = v4r8::load(ptcl[i].jrk);
		v4r8 snp  = v4r8::load(ptcl[i].snp);
		v4r8 crk  = v4r8::load(ptcl[i].crk);
		v4r8 d4a  = v4r8::load(ptcl[i].d4a);
		v4r8 d5a  = v4r8::load(ptcl[i].d5a);
		v4r8 mass = v4r8::load(&ptcl[i].mass);

		pos = pos + dt*(vel + dt2*(acc + dt3*(jrk + dt4*(snp + dt5*(crk + dt6*(d4a + dt7*(d5a)))))));
		vel = vel + dt*(acc + dt2*(jrk + dt3*(snp + dt4*(crk + dt5*(d4a + dt6*(d5a))))));
		acc = acc + dt*(jrk + dt2*(snp + dt3*(crk + dt4*(d4a + dt5*(d5a)))));
		jrk = jrk + dt*(snp + dt2*(crk + dt3*(d4a + dt4*(d5a))));
		pos = pos.ecsl(3).ecsl(mass, 1);

		const v4_mask v3mask(1,1,1,0);
		pos.store(pred[i].pos);
		vel.store(pred[i].vel, v3mask);
		acc.store(pred[i].acc, v3mask);
		jrk.store(pred[i].jrk, v3mask);
	}
	// v4r8::simd_mode_2();
}

void Gravity::calc_force_in_range_fast_omp(
		const int    is,
		const int    ie,
		const double eps2_s,
		Force * __restrict force )
{
	v4r8::simd_mode_4();
	static GForce fobuf[MAXTHREAD][NIMAX/4 + 1];
	const int tid = omp_get_thread_num();
	const int nthreads = omp_get_num_threads();

	const int nj = nbody;

	for(int i=is, ii=0; i<ie; i+=4, ii++){
		v4r8 ax(0.0);
		v4r8 ay(0.0);
		v4r8 az(0.0);
		v4r8 jx(0.0);
		v4r8 jy(0.0);
		v4r8 jz(0.0);
		v4r8 sx(0.0);
		v4r8 sy(0.0);
		v4r8 sz(0.0);
		v4r8 cx(0.0);
		v4r8 cy(0.0);
		v4r8 cz(0.0);

		v4r8 xi = v4r8::load((double *)&pred[i+0].pos);
		v4r8 yi = v4r8::load((double *)&pred[i+1].pos);
		v4r8 zi = v4r8::load((double *)&pred[i+2].pos);
		v4r8 wi = v4r8::load((double *)&pred[i+3].pos);

		v4r8 vxi = v4r8::load((double *)&pred[i+0].vel);
		v4r8 vyi = v4r8::load((double *)&pred[i+1].vel);
		v4r8 vzi = v4r8::load((double *)&pred[i+2].vel);
		v4r8 vwi = v4r8::load((double *)&pred[i+3].vel);

		v4r8 axi = v4r8::load((double *)&pred[i+0].acc);
		v4r8 ayi = v4r8::load((double *)&pred[i+1].acc);
		v4r8 azi = v4r8::load((double *)&pred[i+2].acc);
		v4r8 awi = v4r8::load((double *)&pred[i+3].acc);

		v4r8 jxi = v4r8::load((double *)&pred[i+0].jrk);
		v4r8 jyi = v4r8::load((double *)&pred[i+1].jrk);
		v4r8 jzi = v4r8::load((double *)&pred[i+2].jrk);
		v4r8 jwi = v4r8::load((double *)&pred[i+3].jrk);

		v4r8::transpose(xi, yi, zi, wi);
		v4r8::transpose(vxi, vyi, vzi, vwi);
		v4r8::transpose(axi, ayi, azi, awi);
		v4r8::transpose(jxi, jyi, jzi, jwi);

		v4r8 eps2(eps2_s);

#pragma omp for nowait
		for(int j=0; j<nj; j++){
			const v4r8 xj  = v4r8::broadcastload(&pred[j].pos[0]);
			const v4r8 yj  = v4r8::broadcastload(&pred[j].pos[1]);
			const v4r8 zj  = v4r8::broadcastload(&pred[j].pos[2]);
			const v4r8 mj  = v4r8::broadcastload(&pred[j].mass  );
			const v4r8 vxj = v4r8::broadcastload(&pred[j].vel[0]);
			const v4r8 vyj = v4r8::broadcastload(&pred[j].vel[1]);
			const v4r8 vzj = v4r8::broadcastload(&pred[j].vel[2]);
			const v4r8 axj = v4r8::broadcastload(&pred[j].acc[0]);
			const v4r8 ayj = v4r8::broadcastload(&pred[j].acc[1]);
			const v4r8 azj = v4r8::broadcastload(&pred[j].acc[2]);
			const v4r8 jxj = v4r8::broadcastload(&pred[j].jrk[0]);
			const v4r8 jyj = v4r8::broadcastload(&pred[j].jrk[1]);
			const v4r8 jzj = v4r8::broadcastload(&pred[j].jrk[2]);

			const v4r8 dx = xj - xi;
			const v4r8 dy = yj - yi;
			const v4r8 dz = zj - zi;
			const v4r8 dvx = vxj - vxi;
			const v4r8 dvy = vyj - vyi;
			const v4r8 dvz = vzj - vzi;
			const v4r8 dax = axj - axi;
			const v4r8 day = ayj - ayi;
			const v4r8 daz = azj - azi;
			const v4r8 djx = jxj - jxi;
			const v4r8 djy = jyj - jyi;
			const v4r8 djz = jzj - jzi;

			const v4r8 dr2 = ((eps2 + dx*dx) + dy*dy) + dz*dz;
			const v4r8 drdv = ( dx*dvx +  dy*dvy) + dz*dvz;
			const v4r8 drda = ( dx*dax +  dy*day) + dz *daz;
			const v4r8 dvdv = (dvx*dvx + dvy*dvy) + dvz*dvz;
			const v4r8 dvda = (dvx*dax + dvy*day) + dvz*daz;
			const v4r8 drdj = ( dx*djx +  dy*djy) +  dz*djz;

			const v4r8 rinv1 = dr2.rsqrta_x8();
			const v4r8 rinv2 = rinv1 * rinv1;
			const v4r8 mrinv3 = mj * rinv1 * rinv2;

			v4r8 alpha = drdv * rinv2;
			v4r8 alphaalpha = alpha * alpha;
			v4r8 beta  = (dvdv + drda) * rinv2 + alphaalpha;
			v4r8 gamma = (v4r8(3.0)*dvda + drdj) * rinv2
						+ alpha * (v4r8(3.0)*beta - v4r8(4.0)*alphaalpha);

			ax += mrinv3 * dx;
			ay += mrinv3 * dy;
			az += mrinv3 * dz;

			alpha *= v4r8(-3.0);
			v4r8 tx = dvx + alpha * dx;
			v4r8 ty = dvy + alpha * dy;
			v4r8 tz = dvz + alpha * dz;
			jx += mrinv3 * tx;
			jy += mrinv3 * ty;
			jz += mrinv3 * tz;

			alpha += alpha;
			beta  *= v4r8(-3.0);
			v4r8 ux = (dax + alpha * tx) + beta * dx;
			v4r8 uy = (day + alpha * ty) + beta * dy;
			v4r8 uz = (daz + alpha * tz) + beta * dz;
			sx += mrinv3 * ux;
			sy += mrinv3 * uy;
			sz += mrinv3 * uz;

			alpha *= v4r8(1.5);
			beta  *= v4r8(3.0);
			gamma *= v4r8(-3.0);
			cx += mrinv3 * (((djx + alpha * ux) + beta * tx) + gamma * dx);
			cy += mrinv3 * (((djy + alpha * uy) + beta * ty) + gamma * dy);
			cz += mrinv3 * (((djz + alpha * uz) + beta * tz) + gamma * dz);
		}
		fobuf[tid][ii].save(ax, ay, az, jx, jy, jz, sx, sy, sz, cx, cy, cz);
	} // for(i)
#pragma omp barrier
	if(1){
#pragma omp for
		for(int i=is; i<ie; i+=4){
			const int ii = (i-is)/4;
			v4r8 ax, ay, az, jx, jy, jz, sx, sy, sz, cx, cy, cz;
#pragma loop noswp
#pragma loop unroll 8
			for(int ith=0; ith<nthreads; ith++){
				ax += fobuf[ith][ii].ax;
				ay += fobuf[ith][ii].ay;
				az += fobuf[ith][ii].az;
				jx += fobuf[ith][ii].jx;
				jy += fobuf[ith][ii].jy;
				jz += fobuf[ith][ii].jz;
				sx += fobuf[ith][ii].sx;
				sy += fobuf[ith][ii].sy;
				sz += fobuf[ith][ii].sz;
				cx += fobuf[ith][ii].cx;
				cy += fobuf[ith][ii].cy;
				cz += fobuf[ith][ii].cz;
			}
			GForce(ax, ay, az, jx, jy, jz, sx, sy, sz, cx, cy, cz).store_4_forces(force + i);
		} // here comes a barrier
	}
}

void Gravity::calc_potential_rp(
		const int    nbody,
		const double deps2,
		const GParticle * __restrict ptcl,
		v4r8            * __restrict xmbuf,
		double          * __restrict potbuf)
{
// #pragma omp parallel
	{
		v4r8::simd_mode_4();
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
