// #warning AVX 8th-order
#include <omp.h>

#include "v4df_util.h"

struct Gravity{
	enum{
		NIMAX = 1024,
		MAXTHREAD = 64,
		NACT_PARALLEL_THRESH = 12,
	};

	struct GParticle{
		v4df pos_mass;
		v4df vel_time;
		v4df acc;
		v4df jrk;
		v4df snp;
		v4df crk;
		v4df d4a;
		v4df d5a;
	};

	struct GPredictor{
		v4df pos_mass;
		v4df vel;
		v4df acc;
		v4df jrk;
	};

	struct GForce{
		v4df ax, ay, az;
		v4df jx, jy, jz;
		v4df sx, sy, sz;
		v4df cx, cy, cz;

		void clear(){
			ax = ay = az = (v4df){0.0, 0.0, 0.0, 0.0};
			jx = jy = jz = (v4df){0.0, 0.0, 0.0, 0.0};
			sx = sy = sz = (v4df){0.0, 0.0, 0.0, 0.0};
			cx = cy = cz = (v4df){0.0, 0.0, 0.0, 0.0};
		}

		void accumulate(const GForce &rhs){
			ax += rhs.ax;
			ay += rhs.ay;
			az += rhs.az;
			jx += rhs.jx;
			jy += rhs.jy;
			jz += rhs.jz;
			sx += rhs.sx;
			sy += rhs.sy;
			sz += rhs.sz;
			cx += rhs.cx;
			cy += rhs.cy;
			cz += rhs.cz;
		}

		void save(
				const v4df _ax, const v4df _ay, const v4df _az,
				const v4df _jx, const v4df _jy, const v4df _jz,
				const v4df _sx, const v4df _sy, const v4df _sz,
				const v4df _cx, const v4df _cy, const v4df _cz)
		{
			ax = _ax; ay = _ay; az = _az;
			jx = _jx; jy = _jy; jz = _jz;
			sx = _sx; sy = _sy; sz = _sz;
			cx = _cx; cy = _cy; cz = _cz;
		}

		void store_4_forces(Force fout[]){
			// store 12x4 words, optimized later
			v4df vdum = {0.0, 0.0, 0.0, 0.0};
			v4df_transpose atrans(ax, ay, az, vdum);
			v4df_transpose jtrans(jx, jy, jz, vdum);
			v4df_transpose strans(sx, sy, sz, vdum);
			v4df_transpose ctrans(cx, cy, cz, vdum);

#if 0
			fout[0].acc = v4df_to_dvec3(atrans.c0);
			fout[0].jrk = v4df_to_dvec3(jtrans.c0);
			fout[0].snp = v4df_to_dvec3(strans.c0);
			fout[0].crk = v4df_to_dvec3(ctrans.c0);

			fout[1].acc = v4df_to_dvec3(atrans.c1);
			fout[1].jrk = v4df_to_dvec3(jtrans.c1);
			fout[1].snp = v4df_to_dvec3(strans.c1);
			fout[1].crk = v4df_to_dvec3(ctrans.c1);

			fout[2].acc = v4df_to_dvec3(atrans.c2);
			fout[2].jrk = v4df_to_dvec3(jtrans.c2);
			fout[2].snp = v4df_to_dvec3(strans.c2);
			fout[2].crk = v4df_to_dvec3(ctrans.c2);

			fout[3].acc = v4df_to_dvec3(atrans.c3);
			fout[3].jrk = v4df_to_dvec3(jtrans.c3);
			fout[3].snp = v4df_to_dvec3(strans.c3);
			fout[3].crk = v4df_to_dvec3(ctrans.c3);
#else
			v4df ym0  = v4df_vecalign1(atrans.c0, jtrans.c0);
			v4df ym1  = v4df_vecalign2(jtrans.c0, strans.c0);
			v4df ym2  = v4df_vecalign3(strans.c0, ctrans.c0);
			v4df ym3  = v4df_vecalign1(atrans.c1, jtrans.c1);
			v4df ym4  = v4df_vecalign2(jtrans.c1, strans.c1);
			v4df ym5  = v4df_vecalign3(strans.c1, ctrans.c1);
			v4df ym6  = v4df_vecalign1(atrans.c2, jtrans.c2);
			v4df ym7  = v4df_vecalign2(jtrans.c2, strans.c2);
			v4df ym8  = v4df_vecalign3(strans.c2, ctrans.c2);
			v4df ym9  = v4df_vecalign1(atrans.c3, jtrans.c3);
			v4df ym10 = v4df_vecalign2(jtrans.c3, strans.c3);
			v4df ym11 = v4df_vecalign3(strans.c3, ctrans.c3);
			
			v4df *dst = (v4df *)fout;
			dst[ 0]  = ym0;
			dst[ 1]  = ym1;
			dst[ 2]  = ym2;
			dst[ 3]  = ym3;
			dst[ 4]  = ym4;
			dst[ 5]  = ym5;
			dst[ 6]  = ym6;
			dst[ 7]  = ym7;
			dst[ 8]  = ym8;
			dst[ 9]  = ym9;
			dst[10]  = ym10;
			dst[11]  = ym11;
#endif
		}
		void prefetch() const{
			const double *ptr = (const double *)this;
			__builtin_prefetch(ptr +  0);
			// __builtin_prefetch(ptr +  8);
			__builtin_prefetch(ptr + 16);
			// __builtin_prefetch(ptr + 24);
			__builtin_prefetch(ptr + 32);
			// __builtin_prefetch(ptr + 40);
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
		free(ptcl);
		free(pred);
	}

	void set_jp(const int addr, const Particle &p){
		v4df pos_mass = {p.pos.x, p.pos.y, p.pos.z, p.mass};
		v4df vel_time = {p.vel.x, p.vel.y, p.vel.z, p.tlast};
		v4df acc      = {p.acc.x, p.acc.y, p.acc.z, 0.0};
		v4df jrk      = {p.jrk.x, p.jrk.y, p.jrk.z, 0.0};
		v4df snp      = {p.snp.x, p.snp.y, p.snp.z, 0.0};
		v4df crk      = {p.crk.x, p.crk.y, p.crk.z, 0.0};
		v4df d4a      = {p.d4a.x, p.d4a.y, p.d4a.z, 0.0};
		v4df d5a      = {p.d5a.x, p.d5a.y, p.d5a.z, 0.0};

		ptcl[addr].pos_mass = pos_mass;
		ptcl[addr].vel_time = vel_time;
		ptcl[addr].acc      = acc;
		ptcl[addr].jrk      = jrk;
		ptcl[addr].snp      = snp;
		ptcl[addr].crk      = crk;
		ptcl[addr].d4a      = d4a;
		ptcl[addr].d5a      = d5a;
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
			v4df dt4 = dt * (v4df){1./4., 1./4., 1./4., 0.0};
			v4df dt5 = dt * (v4df){1./5., 1./5., 1./5., 0.0};
			v4df dt6 = dt * (v4df){1./6., 1./6., 1./6., 0.0};
			v4df dt7 = dt * (v4df){1./7., 1./7., 1./7., 0.0};

			const GParticle &p = ptcl[i];
			v4df pos_mass = p.pos_mass + dt * (p.vel_time
					+ dt2 * (p.acc + dt3 * (p.jrk 
							+ dt4 * (p.snp + dt5 * (p.crk
									+ dt6 * (p.d4a + dt7 * (p.d5a)))))));
			v4df vel = p.vel_time + dt * (p.acc + dt2 * (p.jrk 
					+ dt3 * (p.snp + dt4 * (p.crk
							+ dt5 * (p.d4a + dt6 * (p.d5a))))));
			v4df acc = p.acc + dt * (p.jrk + dt2 * (p.snp + dt3 * (p.crk
					+ dt4 * (p.d4a + dt5 * (p.d5a)))));
			v4df jrk = p.jrk + dt * (p.snp + dt2 * (p.crk + dt3 * (p.d4a + dt4 * (p.d5a))));

			pred[i].pos_mass = pos_mass;
			pred[i].vel = vel;
			pred[i].acc = acc;
			pred[i].jrk = jrk;
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
			v4df dt4 = dt * (v4df){1./4., 1./4., 1./4., 0.0};
			v4df dt5 = dt * (v4df){1./5., 1./5., 1./5., 0.0};
			v4df dt6 = dt * (v4df){1./6., 1./6., 1./6., 0.0};
			v4df dt7 = dt * (v4df){1./7., 1./7., 1./7., 0.0};

			const GParticle &p = ptcl[i];
			v4df pos_mass = p.pos_mass + dt * (p.vel_time
					+ dt2 * (p.acc + dt3 * (p.jrk 
							+ dt4 * (p.snp + dt5 * (p.crk
									+ dt6 * (p.d4a + dt7 * (p.d5a)))))));
			v4df vel = p.vel_time + dt * (p.acc + dt2 * (p.jrk 
					+ dt3 * (p.snp + dt4 * (p.crk
							+ dt5 * (p.d4a + dt6 * (p.d5a))))));
			v4df acc = p.acc + dt * (p.jrk + dt2 * (p.snp + dt3 * (p.crk
					+ dt4 * (p.d4a + dt5 * (p.d5a)))));
			v4df jrk = p.jrk + dt * (p.snp + dt2 * (p.crk + dt3 * (p.d4a + dt4 * (p.d5a))));

			pred[i].pos_mass = pos_mass;
			pred[i].vel = vel;
			pred[i].acc = acc;
			pred[i].jrk = jrk;
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
				v4df sx = {0.0, 0.0, 0.0, 0.0};
				v4df sy = {0.0, 0.0, 0.0, 0.0};
				v4df sz = {0.0, 0.0, 0.0, 0.0};
				v4df cx = {0.0, 0.0, 0.0, 0.0};
				v4df cy = {0.0, 0.0, 0.0, 0.0};
				v4df cz = {0.0, 0.0, 0.0, 0.0};
				v4df_transpose tr1(pred[i+0].pos_mass, 
				                   pred[i+1].pos_mass, 
								   pred[i+2].pos_mass, 
								   pred[i+3].pos_mass);
				v4df_transpose tr2(pred[i+0].vel, 
				                   pred[i+1].vel, 
								   pred[i+2].vel, 
								   pred[i+3].vel);
				v4df_transpose tr3(pred[i+0].acc, 
				                   pred[i+1].acc, 
								   pred[i+2].acc, 
								   pred[i+3].acc);
				v4df_transpose tr4(pred[i+0].jrk, 
				                   pred[i+1].jrk, 
								   pred[i+2].jrk, 
								   pred[i+3].jrk);
				const v4df xi = tr1.c0;
				const v4df yi = tr1.c1;
				const v4df zi = tr1.c2;
				const v4df vxi = tr2.c0;
				const v4df vyi = tr2.c1;
				const v4df vzi = tr2.c2;
				const v4df axi = tr3.c0;
				const v4df ayi = tr3.c1;
				const v4df azi = tr3.c2;
				const v4df jxi = tr4.c0;
				const v4df jyi = tr4.c1;
				const v4df jzi = tr4.c2;
				const v4df eps2 = {deps2, deps2, deps2, deps2};
#pragma omp for nowait // calculate partial force
				for(int j=0; j<nj; j++){
					v4df_bcast jbuf1(&pred[j].pos_mass);
					v4df_bcast jbuf2(&pred[j].vel);
					v4df_bcast jbuf3(&pred[j].acc);
					v4df_bcast jbuf4(&pred[j].jrk);
					const v4df xj = jbuf1.e0;
					const v4df yj = jbuf1.e1;
					const v4df zj = jbuf1.e2;
					const v4df mj = jbuf1.e3;
					const v4df vxj = jbuf2.e0;
					const v4df vyj = jbuf2.e1;
					const v4df vzj = jbuf2.e2;
					const v4df axj = jbuf3.e0;
					const v4df ayj = jbuf3.e1;
					const v4df azj = jbuf3.e2;
					const v4df jxj = jbuf4.e0;
					const v4df jyj = jbuf4.e1;
					const v4df jzj = jbuf4.e2;

					const v4df dx = xj - xi;
					const v4df dy = yj - yi;
					const v4df dz = zj - zi;
					const v4df dvx = vxj - vxi;
					const v4df dvy = vyj - vyi;
					const v4df dvz = vzj - vzi;
					const v4df dax = axj - axi;
					const v4df day = ayj - ayi;
					const v4df daz = azj - azi;
					const v4df djx = jxj - jxi;
					const v4df djy = jyj - jyi;
					const v4df djz = jzj - jzi;

					const v4df dr2 = eps2 + dx*dx + dy*dy + dz*dz;
					const v4df drdv =  dx*dvx +  dy*dvy +  dz*dvz;
					const v4df dvdv = dvx*dvx + dvy*dvy + dvz*dvz;
					const v4df drda =  dx*dax +  dy*day +  dz*daz;
					const v4df dvda = dvx*dax + dvy*day + dvz*daz;
					const v4df drdj =  dx*djx +  dy*djy +  dz*djz;

					const v4df rinv1 = v4df_rsqrt(dr2);
					const v4df rinv2 = rinv1 * rinv1;
					const v4df mrinv3 = mj * rinv1 * rinv2;

					const v4df three = {3.0, 3.0, 3.0, 3.0};
					const v4df four  = {4.0, 4.0, 4.0, 4.0};
					v4df alpha = drdv * rinv2;
					v4df beta  = (dvdv + drda) * rinv2 + alpha*alpha;
					v4df gamma = (three*dvda + drdj) * rinv2
								+ alpha * (three*beta - four*alpha*alpha);

					ax += mrinv3 * dx;
					ay += mrinv3 * dy;
					az += mrinv3 * dz;

					alpha *= (v4df){-3.0, -3.0, -3.0, -3.0};
					v4df tx = dvx + alpha * dx;
					v4df ty = dvy + alpha * dy;
					v4df tz = dvz + alpha * dz;
					jx += mrinv3 * tx;
					jy += mrinv3 * ty;
					jz += mrinv3 * tz;

					alpha *= (v4df){2.0, 2.0, 2.0, 2.0};     // -6.0
					beta  *= (v4df){-3.0, -3.0, -3.0, -3.0}; // -3.0
					v4df ux = dax + alpha * tx + beta * dx;
					v4df uy = day + alpha * ty + beta * dy;
					v4df uz = daz + alpha * tz + beta * dz;
					sx += mrinv3 * ux;
					sy += mrinv3 * uy;
					sz += mrinv3 * uz;

					alpha *= (v4df){1.5, 1.5, 1.5, 1.5};     // -9.0
					beta  *= (v4df){3.0, 3.0, 3.0, 3.0};     // -9.0
					gamma *= (v4df){-3.0, -3.0, -3.0, -3.0}; // -3.0
					cx += mrinv3 * (djx + alpha * ux + beta * tx + gamma * dx);
					cy += mrinv3 * (djy + alpha * uy + beta * ty + gamma * dy);
					cz += mrinv3 * (djz + alpha * uz + beta * tz + gamma * dz);
				}
				fobuf[ii].save(ax, ay, az, jx, jy, jz, 
				               sx, sy, sz, cx, cy, cz);
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
#if 0
		// reduction & store
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
			v4df sx = {0.0, 0.0, 0.0, 0.0};
			v4df sy = {0.0, 0.0, 0.0, 0.0};
			v4df sz = {0.0, 0.0, 0.0, 0.0};
			v4df cx = {0.0, 0.0, 0.0, 0.0};
			v4df cy = {0.0, 0.0, 0.0, 0.0};
			v4df cz = {0.0, 0.0, 0.0, 0.0};
			v4df_transpose tr1(pred[i+0].pos_mass, 
					pred[i+1].pos_mass, 
					pred[i+2].pos_mass, 
					pred[i+3].pos_mass);
			v4df_transpose tr2(pred[i+0].vel, 
					pred[i+1].vel, 
					pred[i+2].vel, 
					pred[i+3].vel);
			v4df_transpose tr3(pred[i+0].acc, 
					pred[i+1].acc, 
					pred[i+2].acc, 
					pred[i+3].acc);
			v4df_transpose tr4(pred[i+0].jrk, 
					pred[i+1].jrk, 
					pred[i+2].jrk, 
					pred[i+3].jrk);
#if 0
			const double *ptr = (const double *)(&pred[i+4]);
			__builtin_prefetch(ptr +  0);
			__builtin_prefetch(ptr +  8);
			__builtin_prefetch(ptr + 16);
			__builtin_prefetch(ptr + 24);
			__builtin_prefetch(ptr + 32);
			__builtin_prefetch(ptr + 40);
			__builtin_prefetch(ptr + 48);
			__builtin_prefetch(ptr + 56);
#endif
			const v4df xi = tr1.c0;
			const v4df yi = tr1.c1;
			const v4df zi = tr1.c2;
			const v4df vxi = tr2.c0;
			const v4df vyi = tr2.c1;
			const v4df vzi = tr2.c2;
			const v4df axi = tr3.c0;
			const v4df ayi = tr3.c1;
			const v4df azi = tr3.c2;
			const v4df jxi = tr4.c0;
			const v4df jyi = tr4.c1;
			const v4df jzi = tr4.c2;
			const v4df eps2 = {deps2, deps2, deps2, deps2};
#pragma omp for nowait // calculate partial force
			for(int j=0; j<nj; j++){
				v4df_bcast jbuf1(&pred[j].pos_mass);
				v4df_bcast jbuf2(&pred[j].vel);
				v4df_bcast jbuf3(&pred[j].acc);
				v4df_bcast jbuf4(&pred[j].jrk);
				const v4df xj = jbuf1.e0;
				const v4df yj = jbuf1.e1;
				const v4df zj = jbuf1.e2;
				const v4df mj = jbuf1.e3;
				const v4df vxj = jbuf2.e0;
				const v4df vyj = jbuf2.e1;
				const v4df vzj = jbuf2.e2;
				const v4df axj = jbuf3.e0;
				const v4df ayj = jbuf3.e1;
				const v4df azj = jbuf3.e2;
				const v4df jxj = jbuf4.e0;
				const v4df jyj = jbuf4.e1;
				const v4df jzj = jbuf4.e2;

				const v4df dx = xj - xi;
				const v4df dy = yj - yi;
				const v4df dz = zj - zi;
				const v4df dvx = vxj - vxi;
				const v4df dvy = vyj - vyi;
				const v4df dvz = vzj - vzi;
				const v4df dax = axj - axi;
				const v4df day = ayj - ayi;
				const v4df daz = azj - azi;
				const v4df djx = jxj - jxi;
				const v4df djy = jyj - jyi;
				const v4df djz = jzj - jzi;

				const v4df dr2 = eps2 + dx*dx + dy*dy + dz*dz;
				const v4df drdv =  dx*dvx +  dy*dvy +  dz*dvz;
				const v4df dvdv = dvx*dvx + dvy*dvy + dvz*dvz;
				const v4df drda =  dx*dax +  dy*day +  dz*daz;
				const v4df dvda = dvx*dax + dvy*day + dvz*daz;
				const v4df drdj =  dx*djx +  dy*djy +  dz*djz;

				const v4df rinv1 = v4df_rsqrt(dr2);
				const v4df rinv2 = rinv1 * rinv1;
				const v4df mrinv3 = (mj * rinv1) * rinv2;

				const v4df three = {3.0, 3.0, 3.0, 3.0};
				const v4df four  = {4.0, 4.0, 4.0, 4.0};
				// const v4df frac43  = {4./3., 4./3., 4./3., 4./3.};
				v4df alpha = drdv * rinv2;
				v4df beta  = (dvdv + drda) * rinv2 + alpha*alpha;
				v4df gamma = (three*dvda + drdj) * rinv2 + alpha * (three*beta - four*(alpha*alpha));
				// v4df gamma = (three*dvda + drdj) * rinv2 + (three * alpha) * (beta - frac43*(alpha*alpha));

				ax += mrinv3 * dx;
				ay += mrinv3 * dy;
				az += mrinv3 * dz;

#if 0
				alpha *= (v4df){-3.0, -3.0, -3.0, -3.0};
				v4df tx = dvx + alpha * dx;
				v4df ty = dvy + alpha * dy;
				v4df tz = dvz + alpha * dz;
				jx += mrinv3 * tx;
				jy += mrinv3 * ty;
				jz += mrinv3 * tz;

				alpha *= (v4df){2.0, 2.0, 2.0, 2.0};     // -6.0
				beta  *= (v4df){-3.0, -3.0, -3.0, -3.0}; // -3.0
				v4df ux = dax + alpha * tx + beta * dx;
				v4df uy = day + alpha * ty + beta * dy;
				v4df uz = daz + alpha * tz + beta * dz;
				sx += mrinv3 * ux;
				sy += mrinv3 * uy;
				sz += mrinv3 * uz;

				alpha *= (v4df){1.5, 1.5, 1.5, 1.5};     // -9.0
				beta  *= (v4df){3.0, 3.0, 3.0, 3.0};     // -9.0
				gamma *= (v4df){-3.0, -3.0, -3.0, -3.0}; // -3.0
				cx += mrinv3 * (djx + alpha * ux + beta * tx + gamma * dx);
				cy += mrinv3 * (djy + alpha * uy + beta * ty + gamma * dy);
				cz += mrinv3 * (djz + alpha * uz + beta * tz + gamma * dz);
#else
				alpha *= three;
				v4df tx = dvx - alpha * dx;
				v4df ty = dvy - alpha * dy;
				v4df tz = dvz - alpha * dz;
				jx += mrinv3 * tx;
				jy += mrinv3 * ty;
				jz += mrinv3 * tz;

				// alpha *= (v4df){2.0, 2.0, 2.0, 2.0};     // -6.0
				v4df alpha2 = alpha + alpha;
				beta  *= three; // -3.0
				v4df ux = (dax - alpha2 * tx) - beta * dx;
				v4df uy = (day - alpha2 * ty) - beta * dy;
				v4df uz = (daz - alpha2 * tz) - beta * dz;
				sx += mrinv3 * ux;
				sy += mrinv3 * uy;
				sz += mrinv3 * uz;

				// alpha *= (v4df){1.5, 1.5, 1.5, 1.5};     // -9.0
				v4df alpha3 = alpha2 + alpha;
				beta  *= three;     // -9.0
				gamma *= three; // -3.0
				cx += mrinv3 * (((djx - alpha3 * ux) - beta * tx) - gamma * dx);
				cy += mrinv3 * (((djy - alpha3 * uy) - beta * ty) - gamma * dy);
				cz += mrinv3 * (((djz - alpha3 * uz) - beta * tz) - gamma * dz);
#endif
			}
			fobuf[ii].save(ax, ay, az, jx, jy, jz, 
					sx, sy, sz, cx, cy, cz);
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
