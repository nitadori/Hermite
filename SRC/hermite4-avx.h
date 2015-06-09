#include <omp.h>

#include "v4df_util.h"

struct Gravity{
	enum{
		NIMAX = 2048,
		MAXTHREAD = 64,
		NACT_PARALLEL_THRESH = 8,
	};

	struct GParticle{
		v4df pos_mass;
		v4df vel_time;
		v4df acc;
		v4df jrk;
	};

	struct GPredictor{
		v4df pos_mass;
		v4df vel;
	};

	struct GForce{
		v4df ax, ay, az;
		v4df jx, jy, jz;

		void clear(){
			ax = ay = az = (v4df){0.0, 0.0, 0.0, 0.0};
			jx = jy = jz = (v4df){0.0, 0.0, 0.0, 0.0};
		}

		void accumulate(const GForce &rhs){
			ax += rhs.ax;
			ay += rhs.ay;
			az += rhs.az;
			jx += rhs.jx;
			jy += rhs.jy;
			jz += rhs.jz;
		}

		void save(
				const v4df _ax, const v4df _ay, const v4df _az,
				const v4df _jx, const v4df _jy, const v4df _jz)
		{
			ax = _ax; ay = _ay; az = _az;
			jx = _jx; jy = _jy; jz = _jz;
		}

		static void store_1_force(const v4df acc, const v4df jrk, Force &fout){
			const v2df axay = __builtin_ia32_vextractf128_pd256(acc, 0);
			const v2df az__ = __builtin_ia32_vextractf128_pd256(acc, 1);
			const v2df jxjy = __builtin_ia32_vextractf128_pd256(jrk, 0);
			const v2df jz__ = __builtin_ia32_vextractf128_pd256(jrk, 1);
			const v2df azjx = __builtin_ia32_shufpd(az__, jxjy, 0+(0<<1));
			const v2df jyjz = __builtin_ia32_shufpd(jxjy, jz__, 1+(0<<1));
			v2df *ptr = (v2df *)&fout;
			ptr[0] = axay;
			ptr[1] = azjx;
			ptr[2] = jyjz;
		}

		void store_4_forces(Force fout[]){
			v4df vdum = {0.0, 0.0, 0.0, 0.0};
			v4df_transpose atrans(ax, ay, az, vdum);
			v4df_transpose jtrans(jx, jy, jz, vdum);
#if 0
			fout[0].acc = v4df_to_dvec3(atrans.c0);
			fout[0].jrk = v4df_to_dvec3(jtrans.c0);
			fout[1].acc = v4df_to_dvec3(atrans.c1);
			fout[1].jrk = v4df_to_dvec3(jtrans.c1);
			fout[2].acc = v4df_to_dvec3(atrans.c2);
			fout[2].jrk = v4df_to_dvec3(jtrans.c2);
			fout[3].acc = v4df_to_dvec3(atrans.c3);
			fout[3].jrk = v4df_to_dvec3(jtrans.c3);
#elif 0
			store_1_force(atrans.c0, jtrans.c0, fout[0]);
			store_1_force(atrans.c1, jtrans.c1, fout[1]);
			store_1_force(atrans.c2, jtrans.c2, fout[2]);
			store_1_force(atrans.c3, jtrans.c3, fout[3]);
#else
			v4df ym0  = v4df_vecalign1(atrans.c0, jtrans.c0);
			v4df ym1  = v4df_vecalign2(jtrans.c0, atrans.c1);
			v4df ym2  = v4df_vecalign3(atrans.c1, jtrans.c1);
			v4df ym3  = v4df_vecalign1(atrans.c2, jtrans.c2);
			v4df ym4  = v4df_vecalign2(jtrans.c2, atrans.c3);
			v4df ym5  = v4df_vecalign3(atrans.c3, jtrans.c3);

			v4df *dst = (v4df *)fout;
			dst[0] = ym0;
			dst[1] = ym1;
			dst[2] = ym2;
			dst[3] = ym3;
			dst[4] = ym4;
			dst[5] = ym5;
#endif
		}

		void prefetch() const{
			const double *ptr = (const double *)this;
			__builtin_prefetch(ptr +  0);
			__builtin_prefetch(ptr +  8);
			__builtin_prefetch(ptr + 16);
		}
	};

	const int  nbody;
	GParticle  *ptcl;
	GPredictor *pred;

	Gravity(const int _nbody) : nbody(_nbody) {
		ptcl = allocate<GParticle,  64> (nbody);
		pred = allocate<GPredictor, 64> (nbody);
#pragma omp parallel
		assert(MAXTHREAD >= omp_get_thread_num());
	}

	~Gravity(){
		deallocate<GParticle,  64> (ptcl);
		deallocate<GPredictor, 64> (pred);
	}

	void set_jp(const int addr, const Particle &p){
		v4df pos_mass = {p.pos.x, p.pos.y, p.pos.z, p.mass};
		v4df vel_time = {p.vel.x, p.vel.y, p.vel.z, p.tlast};
		v4df acc      = {p.acc.x, p.acc.y, p.acc.z, 0.0};
		v4df jrk      = {p.jrk.x, p.jrk.y, p.jrk.z, 0.0};

		ptcl[addr].pos_mass = pos_mass;
		ptcl[addr].vel_time = vel_time;
		ptcl[addr].acc      = acc;
		ptcl[addr].jrk      = jrk;
	}

	void predict_all(const double tsys){
#if 0
#pragma omp parallel for
		for(int i=0; i<nbody; i++){
			const double tlast = *((double *)&ptcl[i].vel_time + 3);
			const double dts = tsys - tlast;
			v4df dt = {dts, dts, dts, 0.0};
			v4df dt2 = dt * (v4df){0.5, 0.5, 0.5, 0.0};
			v4df dt3 = dt * (v4df){1./3., 1./3., 1./3., 0.0};
			const GParticle &p = ptcl[i];
			v4df pos_mass = p.pos_mass + dt * (p.vel_time + dt2 * (p.acc + dt3 * (p.jrk)));
			v4df vel = p.vel_time + dt * (p.acc + dt2 * (p.jrk));
			pred[i].pos_mass = pos_mass;
			pred[i].vel = vel;
		}
#else
#pragma omp parallel
		predict_all_fast_omp(tsys);
#endif
	}
	void predict_all_fast_omp(const double tsys){
#pragma omp for nowait
		for(int i=0; i<nbody; i++){
			const double tlast = *((double *)&ptcl[i].vel_time + 3);
			const double dts = tsys - tlast;
			v4df dt = {dts, dts, dts, 0.0};
			v4df dt2 = dt * (v4df){0.5, 0.5, 0.5, 0.0};
			v4df dt3 = dt * (v4df){1./3., 1./3., 1./3., 0.0};
			const GParticle &p = ptcl[i];
			v4df pos_mass = p.pos_mass + dt * (p.vel_time + dt2 * (p.acc + dt3 * (p.jrk)));
			v4df vel = p.vel_time + dt * (p.acc + dt2 * (p.jrk));
			pred[i].pos_mass = pos_mass;
			pred[i].vel = vel;
		}
	}

#if 0
	void calc_force_in_range(
			const int    is,
			const int    ie,
			const double deps2,
			Force        force[] )
	{
		static __thread GForce fobuf[NIMAX/4];
		static GForce *foptr[MAXTHREAD];
		int nthreads;
#pragma omp parallel
		{
			const int tid = omp_get_thread_num();
#pragma omp master
			nthreads = omp_get_num_threads();
			// foptr[tid] = fobuf;
			if(foptr[tid] != fobuf) foptr[tid] = fobuf;
			const int nj = nbody;
			// simiple i-parallel AVX force
			for(int i=is, ii=0; i<ie; i+=4, ii++){
				v4df ax = {0.0, 0.0, 0.0, 0.0};
				v4df ay = {0.0, 0.0, 0.0, 0.0};
				v4df az = {0.0, 0.0, 0.0, 0.0};
				v4df jx = {0.0, 0.0, 0.0, 0.0};
				v4df jy = {0.0, 0.0, 0.0, 0.0};
				v4df jz = {0.0, 0.0, 0.0, 0.0};
				v4df_transpose tr1(pred[i+0].pos_mass, 
				                   pred[i+1].pos_mass, 
								   pred[i+2].pos_mass, 
								   pred[i+3].pos_mass);
				v4df_transpose tr2(pred[i+0].vel, 
				                   pred[i+1].vel, 
								   pred[i+2].vel, 
								   pred[i+3].vel);
				const v4df xi = tr1.c0;
				const v4df yi = tr1.c1;
				const v4df zi = tr1.c2;
				const v4df vxi = tr2.c0;
				const v4df vyi = tr2.c1;
				const v4df vzi = tr2.c2;
				const v4df eps2 = {deps2, deps2, deps2, deps2};
#pragma omp for nowait // calculate partial force
				for(int j=0; j<nj; j++){
					v4df_bcast jbuf1(&pred[j].pos_mass);
					v4df_bcast jbuf2(&pred[j].vel);
					const v4df xj = jbuf1.e0;
					const v4df yj = jbuf1.e1;
					const v4df zj = jbuf1.e2;
					const v4df mj = jbuf1.e3;
					const v4df vxj = jbuf2.e0;
					const v4df vyj = jbuf2.e1;
					const v4df vzj = jbuf2.e2;

					const v4df dx = xj - xi;
					const v4df dy = yj - yi;
					const v4df dz = zj - zi;
					const v4df dvx = vxj - vxi;
					const v4df dvy = vyj - vyi;
					const v4df dvz = vzj - vzi;

					const v4df dr2 = eps2 + dx*dx + dy*dy + dz*dz;
					const v4df drdv = dx*dvx + dy*dvy + dz*dvz;

					const v4df rinv1 = v4df_rsqrt(dr2);
					const v4df rinv2 = rinv1 * rinv1;
					const v4df mrinv3 = mj * rinv1 * rinv2;

					v4df alpha = drdv * rinv2;
					alpha *= (v4df){-3.0, -3.0, -3.0, -3.0};

					ax += mrinv3 * dx;
					ay += mrinv3 * dy;
					az += mrinv3 * dz;
					jx += mrinv3 * (dvx + alpha * dx);
					jy += mrinv3 * (dvy + alpha * dy);
					jz += mrinv3 * (dvz + alpha * dz);
				}
				fobuf[ii].save(ax, ay, az, jx, jy, jz);
			} // for(i)
#pragma omp barrier
#pragma omp for nowait
			for(int i=is; i<ie; i+=4){
				const int ii = (i-is)/4;
				GForce fsum;
				fsum.clear();
				for(int ith=0; ith<nthreads; ith++){
					fsum.accumulate(foptr[ith][ii]);
				}
				fsum.store_4_forces(force + i);
			}
		} // end omp parallel
		// reduction & store
#if 0
		for(int i=is, ii=0; i<ie; i+=4, ii++){
			GForce fsum;
			fsum.clear();
			for(int ith=0; ith<nthreads; ith++){
				fsum.accumulate(foptr[ith][ii]);
			}
			fsum.store_4_forces(force + i);
		}
#endif
	}
#endif
	void calc_force_in_range_fast_omp(
			const int    is,
			const int    ie,
			const double deps2,
			Force        force[] )
	{
		static __thread GForce fobuf[NIMAX/4 + 1];
		static GForce *foptr[MAXTHREAD];
		const int tid = omp_get_thread_num();
		const int nthreads = omp_get_num_threads();
		// foptr[tid] = fobuf;
		if(foptr[tid] != fobuf) foptr[tid] = fobuf;
		const int nj = nbody;
		// simiple i-parallel AVX force
		for(int i=is, ii=0; i<ie; i+=4, ii++){
			v4df ax = {0.0, 0.0, 0.0, 0.0};
			v4df ay = {0.0, 0.0, 0.0, 0.0};
			v4df az = {0.0, 0.0, 0.0, 0.0};
			v4df jx = {0.0, 0.0, 0.0, 0.0};
			v4df jy = {0.0, 0.0, 0.0, 0.0};
			v4df jz = {0.0, 0.0, 0.0, 0.0};
			v4df_transpose tr1(pred[i+0].pos_mass, 
					pred[i+1].pos_mass, 
					pred[i+2].pos_mass, 
					pred[i+3].pos_mass);
			v4df_transpose tr2(pred[i+0].vel, 
					pred[i+1].vel, 
					pred[i+2].vel, 
					pred[i+3].vel);
#if 0
			__builtin_prefetch(&pred[i+4]);
			__builtin_prefetch(&pred[i+5]);
			__builtin_prefetch(&pred[i+6]);
			__builtin_prefetch(&pred[i+7]);
#endif
			const v4df xi = tr1.c0;
			const v4df yi = tr1.c1;
			const v4df zi = tr1.c2;
			const v4df vxi = tr2.c0;
			const v4df vyi = tr2.c1;
			const v4df vzi = tr2.c2;
			const v4df eps2 = {deps2, deps2, deps2, deps2};
#pragma omp for nowait // calculate partial force
			for(int j=0; j<nj; j++){
				v4df_bcast jbuf1(&pred[j].pos_mass);
				v4df_bcast jbuf2(&pred[j].vel);
				const v4df xj = jbuf1.e0;
				const v4df yj = jbuf1.e1;
				const v4df zj = jbuf1.e2;
				const v4df mj = jbuf1.e3;
				const v4df vxj = jbuf2.e0;
				const v4df vyj = jbuf2.e1;
				const v4df vzj = jbuf2.e2;

				const v4df dx = xj - xi;
				const v4df dy = yj - yi;
				const v4df dz = zj - zi;
				const v4df dvx = vxj - vxi;
				const v4df dvy = vyj - vyi;
				const v4df dvz = vzj - vzi;

				const v4df dr2 = eps2 + dx*dx + dy*dy + dz*dz;
				const v4df drdv = dx*dvx + dy*dvy + dz*dvz;

				const v4df rinv1 = v4df_rsqrt(dr2);
				const v4df rinv2 = rinv1 * rinv1;
				const v4df mrinv3 = mj * rinv1 * rinv2;

				v4df alpha = drdv * rinv2;
				alpha *= (v4df){-3.0, -3.0, -3.0, -3.0};

				ax += mrinv3 * dx;
				ay += mrinv3 * dy;
				az += mrinv3 * dz;
				jx += mrinv3 * (dvx + alpha * dx);
				jy += mrinv3 * (dvy + alpha * dy);
				jz += mrinv3 * (dvz + alpha * dz);
			}
			fobuf[ii].save(ax, ay, az, jx, jy, jz);
		} // for(i)
#pragma omp barrier
		if(0 == is && ie-is <= NACT_PARALLEL_THRESH){
#pragma omp master
			for(int i=is; i<ie; i+=4){
				const int ii = (i-is)/4;
				GForce fsum;
				fsum.clear();
				for(int ith=0; ith<nthreads; ith++){
					// burn-out LLC
					foptr[ith][ii].prefetch();
				}
				for(int ith=0; ith<nthreads; ith++){
					fsum.accumulate(foptr[ith][ii]);
				}
				fsum.store_4_forces(force + i);
			} // no wait, return and goto corrector
		}else{
#pragma omp for
			for(int i=is; i<ie; i+=4){
				const int ii = (i-is)/4;
				GForce fsum;
				fsum.clear();
				for(int ith=0; ith<nthreads; ith++){
					// burn-out LLC
					foptr[ith][ii].prefetch();
				}
				for(int ith=0; ith<nthreads; ith++){
					fsum.accumulate(foptr[ith][ii]);
				}
				fsum.store_4_forces(force + i);
			} // here comes a barrier
		}
	}

	void calc_force_on_first_nact(
			const int    nact,
			const double eps2,
			Force        force[] )
	{
#if 0
		for(int ii=0; ii<nact; ii+=NIMAX){
			const int ni = (nact-ii) < NIMAX ? (nact-ii) : NIMAX;
			calc_force_in_range(ii, ii+ni, eps2, force);
		}
#else
#pragma omp parallel
		calc_force_on_first_nact_fast_omp(nact, eps2, force);
#endif
	}
	void calc_force_on_first_nact_fast_omp(
			const int    nact,
			const double eps2,
			Force        force[] )
	{
		if(nact < NIMAX){
			calc_force_in_range_fast_omp(0, nact, eps2, force);
// #pragma omp barrier
		}else{
			for(int ii=0; ii<nact; ii+=NIMAX){
				const int ni = (nact-ii) < NIMAX ? (nact-ii) : NIMAX;
				calc_force_in_range_fast_omp(ii, ii+ni, eps2, force);
// #pragma omp barrier
			}
		}
	}

	void calc_potential(
			const double deps2,
			double       potbuf[] )
	{
		const int ni = nbody;
		const int nj = nbody;
#pragma omp parallel for
		for(int i=0; i<ni; i+=4){
			// simple i-parallel
			v4df pot = {0.0, 0.0, 0.0, 0.0};
			v4df_transpose tr(ptcl[i+0].pos_mass, ptcl[i+1].pos_mass, ptcl[i+2].pos_mass, ptcl[i+3].pos_mass);
			const v4df xi = tr.c0;
			const v4df yi = tr.c1;
			const v4df zi = tr.c2;
			const v4df eps2 = {deps2, deps2, deps2, deps2};
			for(int j=0; j<nj; j++){
				v4df_bcast bc(&ptcl[j].pos_mass);
				const v4df xj = bc.e0;
				const v4df yj = bc.e1;
				const v4df zj = bc.e2;
				const v4df mj = bc.e3;
				const v4df dx = xj - xi;
				const v4df dy = yj - yi;
				const v4df dz = zj - zi;
				const v4df r2 = eps2 + dx*dx + dy*dy + dz*dz;
				const v4df mask = __builtin_ia32_cmppd256(r2, eps2, 12); // NEQ_OQ
				const v4df rinv = v4df_rsqrt(r2);
				const v4df mrinv = __builtin_ia32_andpd256(mj, mask) * rinv;
				pot -= mrinv;
			}
			*(v4df *)(potbuf + i) = pot;
		}
	}
};

