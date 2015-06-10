#include <cassert>
#include <omp.h>
// #include <iostream>
#include <micvec.h>
#include "mic-util.h"
#include "mic-hard.h"

static inline void prefetch_ptcl(const Particle &p){
	const char *addr = (const char *)&p;
	_mm_prefetch(addr +   0, _MM_HINT_T0);
	_mm_prefetch(addr +  64, _MM_HINT_T0);
	_mm_prefetch(addr + 128, _MM_HINT_T0);
	_mm_prefetch(addr + 192, _MM_HINT_T0);
}

struct VParticle{
	__m512d zm0, zm1, zm2, zm3;
	VParticle(const void *ptr){
		const double * vsrc = (const double *)ptr;
		zm0 = loadu_pd(vsrc +  0);
		zm1 = loadu_pd(vsrc +  8);
		zm2 = loadu_pd(vsrc + 16);
		zm3 = loadu_pd(vsrc + 24);
	}
	void store(void *ptr) const {
		double *vdst = (double *)ptr;
		storeu_pd   (vdst +  0, zm0);
		storeu_pd   (vdst +  8, zm1);
		storeu_pd   (vdst + 16, zm2);
		storeu_pd256(vdst + 24, zm3);
	}
};

struct Gravity{
	enum{
	   	NIMAX   = 4*160,
		NIFORCE = (3*NIMAX)/2, // 1.5 zmm per i-particle
		NACT_PARALLEL_THRESH = 16,
	};

	struct GParticle{
		F64vec8 pos_mass;
		F64vec8 vel_time;
		F64vec8 acc;
		F64vec8 jrk;
		F64vec8 snp;
		F64vec8 crk;
	};

	struct GPredictor{
		F64vec8 pos_mass;
		F64vec8 vel;
		F64vec8 acc;
	};

	const int  nbody;
	GParticle  *ptcl;
	GPredictor *pred;

	Gravity(const int _nbody) : nbody(_nbody) {
		assert(0 == nbody%2);
		ptcl = allocate<GParticle,  64> (nbody/2);
		pred = allocate<GPredictor, 64> (nbody/2);
	}

	~Gravity(){
		deallocate<GParticle,  64> (ptcl);
		deallocate<GPredictor, 64> (pred);
	}

	void set_jp(const int addr, const Particle & __restrict p){
		const int ah = addr/2;
		const int al = addr%2;
		double (*ptr)[8] = (double (*)[8])(ptcl + ah);
		ptr[0][4*al + 0] = p.pos.x;
		ptr[0][4*al + 1] = p.pos.y;
		ptr[0][4*al + 2] = p.pos.z;
		ptr[0][4*al + 3] = p.mass;
		ptr[1][4*al + 0] = p.vel.x;
		ptr[1][4*al + 1] = p.vel.y;
		ptr[1][4*al + 2] = p.vel.z;
		ptr[1][4*al + 3] = p.tlast;
		ptr[2][4*al + 0] = p.acc.x;
		ptr[2][4*al + 1] = p.acc.y;
		ptr[2][4*al + 2] = p.acc.z;
		ptr[2][4*al + 3] = 0.0;
		ptr[3][4*al + 0] = p.jrk.x;
		ptr[3][4*al + 1] = p.jrk.y;
		ptr[3][4*al + 2] = p.jrk.z;
		ptr[3][4*al + 3] = 0.0;
		ptr[4][4*al + 0] = p.snp.x;
		ptr[4][4*al + 1] = p.snp.y;
		ptr[4][4*al + 2] = p.snp.z;
		ptr[4][4*al + 3] = 0.0;
		ptr[5][4*al + 0] = p.crk.x;
		ptr[5][4*al + 1] = p.crk.y;
		ptr[5][4*al + 2] = p.crk.z;
		ptr[5][4*al + 3] = 0.0;
	}

	// __attribute__((noinline))
	void predict_all(const double tsys){
#if 0
		F64vec8 tnow(tsys);
		F64vec8 coef(1./2., 1./3., 1./4., 1./5.);
#pragma omp parallel for
		for(int i=0; i<nbody/2; i++){
			const GParticle &p = ptcl[i];
			F64vec8 dt  = (tnow - p.vel_time).dddd();
			F64vec8 dt2 = dt * coef.aaaa();
			F64vec8 dt3 = dt * coef.bbbb();
			F64vec8 dt4 = dt * coef.cccc();
			F64vec8 dt5 = dt * coef.dddd();
			
			F64vec8 dpos = p.vel_time + 
				dt2 * (p.acc + dt3 * (p.jrk + dt4 * (p.snp + dt5 * p.crk)));
			F64vec8 posm = _mm512_mask3_fmadd_pd(dt, dpos, p.pos_mass, 0x77);
			F64vec8 vel = p.vel_time + 
				dt * (p.acc + dt2 * (p.jrk + dt3 * (p.snp + dt4 * p.crk)));
			F64vec8 acc = p.acc + dt * (p.jrk + dt2 * (p.snp + dt3 * p.crk));

			pred[i].pos_mass = posm;
			pred[i].vel = vel;
			pred[i].acc = acc;
		}
#else
#pragma omp parallel
		predict_all_fast_omp(tsys);
#endif
	}
	__attribute__((noinline))
	void predict_all_fast_omp(const double tsys){
		F64vec8 tnow(tsys);
		F64vec8 coef(1./2., 1./3., 1./4., 1./5.);
#pragma omp for nowait
		for(int i=0; i<nbody/2; i++){
			const GParticle &p = ptcl[i];
			F64vec8 dt  = (tnow - p.vel_time).dddd();
			F64vec8 dt2 = dt * coef.aaaa();
			F64vec8 dt3 = dt * coef.bbbb();
			F64vec8 dt4 = dt * coef.cccc();
			F64vec8 dt5 = dt * coef.dddd();
			
			F64vec8 dpos = p.vel_time + 
				dt2 * (p.acc + dt3 * (p.jrk + dt4 * (p.snp + dt5 * p.crk)));
			F64vec8 posm = _mm512_mask3_fmadd_pd(dt, dpos, p.pos_mass, 0x77);
			F64vec8 vel = p.vel_time + 
				dt * (p.acc + dt2 * (p.jrk + dt3 * (p.snp + dt4 * p.crk)));
			F64vec8 acc = p.acc + dt * (p.jrk + dt2 * (p.snp + dt3 * p.crk));

			pred[i].pos_mass = posm;
			pred[i].vel = vel;
			pred[i].acc = acc;
		}
	}

#if 0
	__attribute__((noinline))
	void calc_force_in_range(
			const int    is,
			const int    ni,
			const double deps2,
			Force        force[] )
	{
		const int nbody = this->nbody;
		static __m512d fpart[NIFORCE][MIC_NCORE];
#pragma omp parallel
		{
			const int nth = omp_get_num_threads();
			const int tid = omp_get_thread_num();
			assert(MIC_NTHRE == nth);
			const int xid = tid % MIC_NSMT;
			const int yid = tid / MIC_NSMT;

			const int iibeg = aligned_division(ni, xid+0, MIC_NSMT, 4);
			const int iiend = aligned_division(ni, xid+1, MIC_NSMT, 4);

			const int jbeg = aligned_division(nbody, yid+0, MIC_NCORE, 2);
			const int jend = aligned_division(nbody, yid+1, MIC_NCORE, 2);

			const F64vec8 eps2(deps2);
			F64vec8 coef(-3.0, 1.0, 0.5, 0.375);

			for(int ii=iibeg; ii<iiend; ii+=4){
				const int iii = is + ii;
				const double *ptr0 = (double *)(&pred[iii/2+0]);
				const double *ptr1 = (double *)(&pred[iii/2+1]);

				F64vec8 xi = _mm512_extload_pd(
						ptr0+ 0, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
				F64vec8 yi = _mm512_extload_pd(
						ptr0+ 4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
				F64vec8 zi = _mm512_extload_pd(
						ptr1+ 0, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
				F64vec8 mi = _mm512_extload_pd(
						ptr1+ 4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
				transpose_4zmm_pd(xi, yi, zi, mi);

				F64vec8 vxi = _mm512_extload_pd(
						ptr0+ 8, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
				F64vec8 vyi = _mm512_extload_pd(
						ptr0+12, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
				F64vec8 vzi = _mm512_extload_pd(
						ptr1+ 8, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
				F64vec8 pad = _mm512_extload_pd(
						ptr1+12, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
				transpose_4zmm_pd(vxi, vyi, vzi, pad);

				F64vec8 axi = _mm512_extload_pd(
						ptr0+16, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
				F64vec8 ayi = _mm512_extload_pd(
						ptr0+20, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
				F64vec8 azi = _mm512_extload_pd(
						ptr1+16, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
				F64vec8 awi = _mm512_extload_pd(
						ptr1+20, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
				transpose_4zmm_pd(axi, ayi, azi, awi);


				F64vec8 ax(0.0), ay(0.0), az(0.0);
				F64vec8 jx(0.0), jy(0.0), jz(0.0);
				F64vec8 sx(0.0), sy(0.0), sz(0.0);

				for(int j=jbeg; j<jend; j+=2){
					const double *jptr = (double *)(&pred[j/2]);
					_mm_prefetch((char *)(jptr + 24), _MM_HINT_T0);
					_mm_prefetch((char *)(jptr + 32), _MM_HINT_T0);
					_mm_prefetch((char *)(jptr + 40), _MM_HINT_T0);
					F64vec8 jxbuf = *(__m512d *)(jptr + 0);
					F64vec8 jvbuf = *(__m512d *)(jptr + 8);
					F64vec8 jabuf = *(__m512d *)(jptr +16);

					const F64vec8 dx  = -(xi  - jxbuf.aaaa());
					const F64vec8 dy  = -(yi  - jxbuf.bbbb());
					const F64vec8 dz  = -(zi  - jxbuf.cccc());
					const F64vec8 dvx = -(vxi - jvbuf.aaaa());
					const F64vec8 dvy = -(vyi - jvbuf.bbbb());
					const F64vec8 dvz = -(vzi - jvbuf.cccc());
					const F64vec8 dax = -(axi - jabuf.aaaa());
					const F64vec8 day = -(ayi - jabuf.bbbb());
					const F64vec8 daz = -(azi - jabuf.cccc());

					const F64vec8 r2 = dz*dz + (dy*dy + (dx*dx + eps2));
					const F64vec8 drdv =  dz*dvz + ( dy*dvy + ( dx*dvx));
					const F64vec8 dvdv = dvz*dvz + (dvy*dvy + (dvx*dvx));
					const F64vec8 drda =  dz*daz + ( dy*day + ( dx*dax));

					const F64vec8 rinv   = rsqrt_pd_x3(
							r2, coef.bbbb(), coef.cccc(), coef.dddd());
					const F64vec8 rinv2  = rinv * rinv;
					const F64vec8 mrinv  = rinv * jxbuf.dddd();
					const F64vec8 mrinv3 = mrinv * rinv2;

					F64vec8 alpha = drdv * rinv2;
					F64vec8 beta  = (dvdv + drda) * rinv2 + alpha * alpha;

					ax  += mrinv3 * dx;
					ay  += mrinv3 * dy;
					az  += mrinv3 * dz;

					alpha *= coef.aaaa();
					F64vec8 tx = dvx + alpha * dx;
					F64vec8 ty = dvy + alpha * dy;
					F64vec8 tz = dvz + alpha * dz;
					jx += mrinv3 * tx;
					jy += mrinv3 * ty;
					jz += mrinv3 * tz;

					alpha += alpha;
					beta  *= coef.aaaa();

					sx += mrinv3 * (dax + alpha * tx + beta * dx);
					sy += mrinv3 * (day + alpha * ty + beta * dy);
					sz += mrinv3 * (daz + alpha * tz + beta * dz);
				} // for(j)
				F64vec8 aw(0.0);
				F64vec8 jw(0.0);
				F64vec8 sw(0.0);

				transpose_4zmm_pd(ax, ay, az, aw);
				transpose_4zmm_pd(jx, jy, jz, jw);
				transpose_4zmm_pd(sx, sy, sz, sw);

				F64vec8 v0 = hadd_2f256(ax, jx);
				F64vec8 v1 = hadd_2f256(sx, ay);
				F64vec8 v2 = hadd_2f256(jy, sy);
				F64vec8 v3 = hadd_2f256(az, jz);
				F64vec8 v4 = hadd_2f256(sz, aw);
				F64vec8 v5 = hadd_2f256(jw, sw);

				const int faddr = (3*ii)/2;
				_mm512_storenrngo_pd(&fpart[faddr+0][yid], v0);
				_mm512_storenrngo_pd(&fpart[faddr+1][yid], v1);
				_mm512_storenrngo_pd(&fpart[faddr+2][yid], v2);
				_mm512_storenrngo_pd(&fpart[faddr+3][yid], v3);
				_mm512_storenrngo_pd(&fpart[faddr+4][yid], v4);
				_mm512_storenrngo_pd(&fpart[faddr+5][yid], v5);
			} // for(ii)
		} // omp parallel
		const int nif = (3*ni)/2;
#pragma omp parallel for
		for(int ii=0; ii<nif; ii++){
			F64vec8 sum(0.0);
			for(int j=0; j<MIC_NCORE; j++){
				sum += F64vec8(fpart[ii][j]);
			}
			((__m512d *)(&force[is]))[ii] = sum;
		} // omp parallel for
	}
#endif
	__attribute__((noinline))
	void calc_force_in_range_fast_omp(
			const int    is,
			const int    ni,
			const double deps2,
			Force        force[] )
	{
		const int nbody = this->nbody;
		static __m512d fpart[NIFORCE][MIC_NCORE];

		const int nth = omp_get_num_threads();
		const int tid = omp_get_thread_num();
		assert(MIC_NTHRE == nth);
		const int xid = tid % MIC_NSMT;
		const int yid = tid / MIC_NSMT;

		const int iibeg = aligned_division(ni, xid+0, MIC_NSMT, 4);
		const int iiend = aligned_division(ni, xid+1, MIC_NSMT, 4);

		const int jbeg = aligned_division(nbody, yid+0, MIC_NCORE, 2);
		const int jend = aligned_division(nbody, yid+1, MIC_NCORE, 2);

		const F64vec8 eps2(deps2);
		F64vec8 coef(-3.0, 1.0, 0.5, 0.375);

		for(int ii=iibeg; ii<iiend; ii+=4){
			const int iii = is + ii;
			const double *ptr0 = (double *)(&pred[iii/2+0]);
			const double *ptr1 = (double *)(&pred[iii/2+1]);

			F64vec8 xi = _mm512_extload_pd(
					ptr0+ 0, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			F64vec8 yi = _mm512_extload_pd(
					ptr0+ 4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			F64vec8 zi = _mm512_extload_pd(
					ptr1+ 0, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			F64vec8 mi = _mm512_extload_pd(
					ptr1+ 4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			transpose_4zmm_pd(xi, yi, zi, mi);

			F64vec8 vxi = _mm512_extload_pd(
					ptr0+ 8, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			F64vec8 vyi = _mm512_extload_pd(
					ptr0+12, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			F64vec8 vzi = _mm512_extload_pd(
					ptr1+ 8, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			F64vec8 pad = _mm512_extload_pd(
					ptr1+12, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			transpose_4zmm_pd(vxi, vyi, vzi, pad);

			F64vec8 axi = _mm512_extload_pd(
					ptr0+16, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			F64vec8 ayi = _mm512_extload_pd(
					ptr0+20, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			F64vec8 azi = _mm512_extload_pd(
					ptr1+16, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			F64vec8 awi = _mm512_extload_pd(
					ptr1+20, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			transpose_4zmm_pd(axi, ayi, azi, awi);


			F64vec8 ax(0.0), ay(0.0), az(0.0);
			F64vec8 jx(0.0), jy(0.0), jz(0.0);
			F64vec8 sx(0.0), sy(0.0), sz(0.0);

			for(int j=jbeg; j<jend; j+=2){
				const double *jptr = (double *)(&pred[j/2]);
				_mm_prefetch((char *)(jptr + 24), _MM_HINT_T0);
				_mm_prefetch((char *)(jptr + 32), _MM_HINT_T0);
				_mm_prefetch((char *)(jptr + 40), _MM_HINT_T0);
				F64vec8 jxbuf = *(__m512d *)(jptr + 0);
				F64vec8 jvbuf = *(__m512d *)(jptr + 8);
				F64vec8 jabuf = *(__m512d *)(jptr +16);

				const F64vec8 dx  = -(xi  - jxbuf.aaaa());
				const F64vec8 dy  = -(yi  - jxbuf.bbbb());
				const F64vec8 dz  = -(zi  - jxbuf.cccc());
				const F64vec8 dvx = -(vxi - jvbuf.aaaa());
				const F64vec8 dvy = -(vyi - jvbuf.bbbb());
				const F64vec8 dvz = -(vzi - jvbuf.cccc());
				const F64vec8 dax = -(axi - jabuf.aaaa());
				const F64vec8 day = -(ayi - jabuf.bbbb());
				const F64vec8 daz = -(azi - jabuf.cccc());

				const F64vec8 r2 = dz*dz + (dy*dy + (dx*dx + eps2));
				const F64vec8 drdv =  dz*dvz + ( dy*dvy + ( dx*dvx));
				const F64vec8 dvdv = dvz*dvz + (dvy*dvy + (dvx*dvx));
				const F64vec8 drda =  dz*daz + ( dy*day + ( dx*dax));

				const F64vec8 rinv   = rsqrt_pd_x3(
						r2, coef.bbbb(), coef.cccc(), coef.dddd());
				const F64vec8 rinv2  = rinv * rinv;
				const F64vec8 mrinv  = rinv * jxbuf.dddd();
				const F64vec8 mrinv3 = mrinv * rinv2;

				F64vec8 alpha = drdv * rinv2;
				F64vec8 beta  = (dvdv + drda) * rinv2 + alpha * alpha;

				ax  += mrinv3 * dx;
				ay  += mrinv3 * dy;
				az  += mrinv3 * dz;

				alpha *= coef.aaaa();
				F64vec8 tx = dvx + alpha * dx;
				F64vec8 ty = dvy + alpha * dy;
				F64vec8 tz = dvz + alpha * dz;
				jx += mrinv3 * tx;
				jy += mrinv3 * ty;
				jz += mrinv3 * tz;

				alpha += alpha;
				beta  *= coef.aaaa();

				sx += mrinv3 * (dax + alpha * tx + beta * dx);
				sy += mrinv3 * (day + alpha * ty + beta * dy);
				sz += mrinv3 * (daz + alpha * tz + beta * dz);
			} // for(j)
			F64vec8 aw(0.0);
			F64vec8 jw(0.0);
			F64vec8 sw(0.0);

			transpose_4zmm_pd(ax, ay, az, aw);
			transpose_4zmm_pd(jx, jy, jz, jw);
			transpose_4zmm_pd(sx, sy, sz, sw);

			F64vec8 v0 = hadd_2f256(ax, jx);
			F64vec8 v1 = hadd_2f256(sx, ay);
			F64vec8 v2 = hadd_2f256(jy, sy);
			F64vec8 v3 = hadd_2f256(az, jz);
			F64vec8 v4 = hadd_2f256(sz, aw);
			F64vec8 v5 = hadd_2f256(jw, sw);

			const int faddr = (3*ii)/2;
			_mm512_storenrngo_pd(&fpart[faddr+0][yid], v0);
			_mm512_storenrngo_pd(&fpart[faddr+1][yid], v1);
			_mm512_storenrngo_pd(&fpart[faddr+2][yid], v2);
			_mm512_storenrngo_pd(&fpart[faddr+3][yid], v3);
			_mm512_storenrngo_pd(&fpart[faddr+4][yid], v4);
			_mm512_storenrngo_pd(&fpart[faddr+5][yid], v5);
		} // for(ii)
#pragma omp barrier

		const int nif = (3*ni)/2;
		if(0 == is && ni <= NACT_PARALLEL_THRESH){
#pragma omp master
			for(int ii=0; ii<nif; ii++){
				F64vec8 sum(0.0);
				for(int j=0; j<MIC_NCORE; j++){
					sum += F64vec8(fpart[ii][j]);
				}
				((__m512d *)(&force[is]))[ii] = sum;
			}
		}else{
#pragma omp for
			for(int ii=0; ii<nif; ii++){
				F64vec8 sum(0.0);
				for(int j=0; j<MIC_NCORE; j++){
					sum += F64vec8(fpart[ii][j]);
				}
				((__m512d *)(&force[is]))[ii] = sum;
			} // omp for
		}
	}

	void calc_force_on_first_nact(
			const int    nact,
			const double eps2,
			Force        force[] )
	{
#if 0
		for(int i=0; i<nact; i+=NIMAX){
			int ni = (nact-i) < NIMAX ? (nact-i) : NIMAX;
			if(ni%4) ni += (4-ni%4);
			calc_force_in_range(i, ni, eps2, force);
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
		for(int i=0; i<nact; i+=NIMAX){
			int ni = (nact-i) < NIMAX ? (nact-i) : NIMAX;
			if(ni%4) ni += (4-ni%4);
			calc_force_in_range_fast_omp(i, ni, eps2, force);
		}
	}

	__attribute__((noinline))
	void calc_potential(
			const double deps2,
			double       potbuf[] )
	{
		const F64vec8 eps2(deps2);
#pragma omp parallel for
		for(int i=0; i<nbody; i+=4){
			const double *ptr0 = (double *)(&ptcl[i/2+0]);
			const double *ptr1 = (double *)(&ptcl[i/2+1]);
			F64vec8 xi = _mm512_extload_pd(
					ptr0+0, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			F64vec8 yi = _mm512_extload_pd(
					ptr0+4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			F64vec8 zi = _mm512_extload_pd(
					ptr1+0, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			F64vec8 mi = _mm512_extload_pd(
					ptr1+4, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
			transpose_4zmm_pd(xi, yi, zi, mi);

			F64vec8 pot(0.0);
			F64vec8 coef(-3.0, 1.0, 0.5, 0.375);

			for(int j=0; j<nbody; j+=2){
				_mm_prefetch((char *)(&ptcl[j/2+1]), _MM_HINT_T0);
				F64vec8 jxbuf = *(__m512d *)(&ptcl[j/2]);

				const F64vec8 dx  = -(xi  - jxbuf.aaaa());
				const F64vec8 dy  = -(yi  - jxbuf.bbbb());
				const F64vec8 dz  = -(zi  - jxbuf.cccc());

				const F64vec8 r2 = dz*dz + (dy*dy + (dx*dx + eps2));
				const F64vec8 rinv = rsqrt_pd_x3(r2, coef.bbbb(), coef.cccc(), coef.dddd());
				__mmask8 mask = _mm512_cmplt_pd_mask(eps2, r2);
				// mask = 0xff;
				pot = _mm512_mask3_fmadd_pd(jxbuf.dddd(), rinv, pot, mask);
			}
			pot = -pot;
			pot += permute4f128(pot, _MM_PERM_BADC);
			if(i%8){ // not aligned
				_mm512_mask_store_pd(potbuf+i-4, 0xf0, pot);
			}else{   // aligned
				_mm512_mask_store_pd(potbuf+i+0, 0x0f, pot);
			}
		}
	}
};
