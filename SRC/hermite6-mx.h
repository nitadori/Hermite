#pragma once
#include<cassert>
#include <omp.h>
#include "mx-v4r8.h"

struct Gravity{
	enum{
		NIMAX = 1024,
		MAXTHREAD = 32,
		NACT_PARALLEL_THRESH = 4,
	};

	struct GParticle{
		double pos[3];
		double mass;
		double vel[3];
		double tlast;
		double acc[3];
		double jrk[3];
		double snp[3];
		double crk[3]; // 5 YMM
	};

	struct GPredictor{
		double pos[3];
		double mass;
		double vel[3];
		double acc[3]; // 5 XMM
	};

	struct GForce{
		v4r8 ax, ay, az;
		v4r8 jx, jy, jz;
		v4r8 sx, sy, sz;

		GForce(){}
		GForce(	
				const v4r8 _ax, const v4r8 _ay, const v4r8 _az,
				const v4r8 _jx, const v4r8 _jy, const v4r8 _jz,
				const v4r8 _sx, const v4r8 _sy, const v4r8 _sz)
			: ax(_ax), ay(_ay), az(_az), 
			  jx(_jx), jy(_jy), jz(_jz),
			  sx(_sx), sy(_sy), sz(_sz) {}


		void clear(){
			ax = ay = az = v4r8(0.0);
			jx = jy = jz = v4r8(0.0);
			sx = sy = sz = v4r8(0.0);
		}

		void save(
				const v4r8 _ax, const v4r8 _ay, const v4r8 _az,
				const v4r8 _jx, const v4r8 _jy, const v4r8 _jz,
				const v4r8 _sx, const v4r8 _sy, const v4r8 _sz)
		{
			ax = _ax; ay = _ay; az = _az;
			jx = _jx; jy = _jy; jz = _jz;
			sx = _sx; sy = _sy; sz = _sz;
		}

		__attribute__((always_inline))
		void store_4_forces(Force * __restrict fout){
			const v4_mask v3mask(1,1,1,0);
			v4r8 a0 = ax;
			v4r8 a1 = ay;
			v4r8 a2 = az;
			v4r8 a3;

			v4r8 j0 = jx;
			v4r8 j1 = jy;
			v4r8 j2 = jz;
			v4r8 j3;

			v4r8 s0 = sx;
			v4r8 s1 = sy;
			v4r8 s2 = sz;
			v4r8 s3;

			v4r8::transpose(a0, a1, a2, a3);
			v4r8::transpose(j0, j1, j2, j3);
			v4r8::transpose(s0, s1, s2, s3);

			a0.store(&fout[0].acc.x, v3mask);
			j0.store(&fout[0].jrk.x, v3mask);
			s0.store(&fout[0].snp.x, v3mask);

			a1.store(&fout[1].acc.x, v3mask);
			j1.store(&fout[1].jrk.x, v3mask);
			s1.store(&fout[1].snp.x, v3mask);

			a2.store(&fout[2].acc.x, v3mask);
			j2.store(&fout[2].jrk.x, v3mask);
			s2.store(&fout[2].snp.x, v3mask);

			a3.store(&fout[3].acc.x, v3mask);
			j3.store(&fout[3].jrk.x, v3mask);
			s3.store(&fout[3].snp.x, v3mask);
		}
	};

	const int  nbody;
	GParticle  *ptcl;
	GPredictor *pred;

	Gravity(const int _nbody) : nbody(_nbody) {
		ptcl = allocate<GParticle,  256> (nbody);
		pred = allocate<GPredictor, 256> (nbody);
#pragma omp parallel
		{
			v4r8::simd_mode_4();
			assert(MAXTHREAD >= omp_get_thread_num());
		}
	}
	~Gravity(){
		deallocate<GParticle,  256> (ptcl);
		deallocate<GPredictor, 256> (pred);
	}

	static void set_jp_rp(
			const int addr,
			const Particle * __restrict p,
			GParticle      * __restrict ptcl)
	{
#if 1
		v4r8::simd_mode_4();

		v4r8 mt   = v4r8::load(&p->mass);
		v4r8 posm = v4r8::load(&p->dt);
		v4r8 velt = v4r8::load(&p->pos.z);
		v4r8 ym2  = v4r8::load(&p->acc.x);
		v4r8 ym3  = v4r8::load(&p->jrk.y);
		v4r8 ym4  = v4r8::load(&p->snp.z);

		posm = posm.ecsl(mt, 1);
		velt = velt.ecsl(mt.ecsl(1), 1);

		double *ptr = ptcl[addr].pos;
		posm.store(ptr +  0);
		velt.store(ptr +  4);
		ym2 .store(ptr +  8);
		ym3 .store(ptr + 12);
		ym4 .store(ptr + 16);
		// ptcl[addr].tlast = p->tlast;
#else
		ptcl[addr].pos[0] = p->pos.x;
		ptcl[addr].pos[1] = p->pos.y;
		ptcl[addr].pos[2] = p->pos.z;
		ptcl[addr].mass   = p->mass;
		ptcl[addr].vel[0] = p->vel.x;
		ptcl[addr].vel[1] = p->vel.y;
		ptcl[addr].vel[2] = p->vel.z;
		ptcl[addr].tlast  = p->tlast;
		ptcl[addr].acc[0] = p->acc.x;
		ptcl[addr].acc[1] = p->acc.y;
		ptcl[addr].acc[2] = p->acc.z;
		ptcl[addr].jrk[0] = p->jrk.x;
		ptcl[addr].jrk[1] = p->jrk.y;
		ptcl[addr].jrk[2] = p->jrk.z;
		ptcl[addr].snp[0] = p->snp.x;
		ptcl[addr].snp[1] = p->snp.y;
		ptcl[addr].snp[2] = p->snp.z;
		ptcl[addr].crk[0] = p->crk.x;
		ptcl[addr].crk[1] = p->crk.y;
		ptcl[addr].crk[2] = p->crk.z;
#endif
	}
	void set_jp(const int addr, const Particle &p){
		set_jp_rp(addr, &p, ptcl);
	}

	static void predict_all_rp_fast_omp(
			const int nbody, 
			const double tsys, 
			const GParticle * __restrict ptcl,
			GPredictor      * __restrict pred);

	void predict_all(const double tsys){
#pragma omp parallel
		predict_all_rp_fast_omp(nbody, tsys, ptcl, pred);
	}
	void predict_all_fast_omp(const double tsys){
		predict_all_rp_fast_omp(nbody, tsys, ptcl, pred);
	}

	void calc_force_in_range_fast_omp(
			const int    is,
			const int    ie,
			const double deps2,
			Force        force[] );

	void calc_force_on_first_nact(
			const int    nact,
			const double eps2,
			Force        force[] )
	{
#pragma omp parallel
		calc_force_on_first_nact_fast_omp(nact, eps2, force);
	}
	void calc_force_on_first_nact_fast_omp(
			const int    nact,
			const double eps2,
			Force        force[] )
	{
		for(int ii=0; ii<nact; ii+=NIMAX){
			const int ni = (nact-ii) < NIMAX ? (nact-ii) : NIMAX;
			calc_force_in_range_fast_omp(ii, ii+ni, eps2, force);
		}
	}

	static void calc_potential_rp(
			const int    nbody,
			const double deps2,
			const GParticle * __restrict ptcl,
			v4r8            * __restrict xmbuf,
			double          * __restrict potbuf);
	void calc_potential(
			const double deps2,
			double * __restrict potbuf)
	{
		// puts("MX calc potential");
		v4r8 *xmbuf = allocate<v4r8, 128>(nbody+1);
#pragma omp parallel
		calc_potential_rp(nbody, deps2, ptcl, xmbuf, potbuf);
		free(xmbuf);
	}

};

