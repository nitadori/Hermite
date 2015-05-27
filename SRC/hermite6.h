#include <cassert>
#include "pownth.h"

struct Force{
#ifdef __MIC__
	dvec3  acc;
	double apad;
	dvec3  jrk;
	double jpad;
	dvec3  snp;
	double spad;
#else
	dvec3 acc;
	dvec3 jrk;
	dvec3 snp;
#endif
};

struct Particle{
	enum {
		order = 6,
		flops = 97,
	};
	long  id;
	double mass;
	double tlast;
	double dt;
	dvec3 pos;
	dvec3 vel;
	dvec3 acc;
	dvec3 jrk;
	dvec3 snp;
	dvec3 crk;
	dvec3 d4a; // save all the derivatives
	dvec3 d5a; // 28 DP words

	Particle(
			const long    _id, 
			const double  _mass,
			const dvec3 & _pos,
			const dvec3 & _vel)
		: id(_id), mass(_mass), tlast(0.0), dt(0.0), 
		  pos(_pos), vel(_vel), 
		  acc(0.0), jrk(0.0), snp(0.0), crk(0.0)
	{
		assert(sizeof(Particle) == 224);
	}

	void dump(FILE *fp){
		fprintf(fp, "%8ld %A", id, mass);
		fprintf(fp, "   %A %A %A", pos.x, pos.y, pos.z);
		fprintf(fp, "   %A %A %A", vel.x, vel.y, vel.z);
		fprintf(fp, "   %A %A %A", acc.x, acc.y, acc.z);
		fprintf(fp, "   %A %A %A", jrk.x, jrk.y, jrk.z);
		fprintf(fp, "   %A %A %A", snp.x, snp.y, snp.z);
		fprintf(fp, "   %A %A %A", crk.x, crk.y, crk.z);
		fprintf(fp, "   %A %A %A", d4a.x, d4a.y, d4a.z);
		fprintf(fp, "   %A %A %A", d5a.x, d5a.y, d5a.z);
		fprintf(fp, "\n");
	}
	void restore(FILE *fp){
		int nread = 0;
		nread += fscanf(fp, "%ld %lA", &id, &mass);
		nread += fscanf(fp, "%lA %lA %lA", &pos.x, &pos.y, &pos.z);
		nread += fscanf(fp, "%lA %lA %lA", &vel.x, &vel.y, &vel.z);
		nread += fscanf(fp, "%lA %lA %lA", &acc.x, &acc.y, &acc.z);
		nread += fscanf(fp, "%lA %lA %lA", &jrk.x, &jrk.y, &jrk.z);
		nread += fscanf(fp, "%lA %lA %lA", &snp.x, &snp.y, &snp.z);
		nread += fscanf(fp, "%lA %lA %lA", &crk.x, &crk.y, &crk.z);
		nread += fscanf(fp, "%lA %lA %lA", &d4a.x, &d4a.y, &d4a.z);
		nread += fscanf(fp, "%lA %lA %lA", &d5a.x, &d5a.y, &d5a.z);
		assert(26 == nread);
	}

	void assign_force(const Force &f){
		acc = f.acc;
		jrk = f.jrk;
		snp = f.snp;
	}

	void init_dt(const double eta_s, const double dtmax){
		const double dtnat = eta_s * sqrt(sqrt(acc.norm2() / snp.norm2()));
		dt = 0.25 * dtmax;
		while(dt > dtnat) dt *= 0.5;
	}

	static double aarseth_step_quant(
			const dvec3 &a0, 
			const dvec3 &a1, 
			const dvec3 &a2, 
			const dvec3 &a3, 
			const dvec3 &a4, 
			const dvec3 &a5, 
			const double etapow)
	{
		double s0 = a0.norm2();
		double s1 = a1.norm2();
		double s2 = a2.norm2();
		double s3 = a3.norm2();
		double s4 = a4.norm2();
		double s5 = a5.norm2();

		double u = sqrt(s0*s2) + s1;
		double l = sqrt(s3*s5) + s4;

		const double dtpow = etapow * (u/l);
		return pow_one_nth_quant<6>(dtpow);
	}
	static double aarseth_step(
			const dvec3 &a0, 
			const dvec3 &a1, 
			const dvec3 &a2, 
			const dvec3 &a3, 
			const dvec3 &a4, 
			const dvec3 &a5, 
			const double eta)
	{
		double s0 = a0.norm2();
		double s1 = a1.norm2();
		double s2 = a2.norm2();
		double s3 = a3.norm2();
		double s4 = a4.norm2();
		double s5 = a5.norm2();

		double u = sqrt(s0*s2) + s1;
		double l = sqrt(s3*s5) + s4;
		return eta * pow(u/l, 1./6.);
	}

	void recalc_dt(const double eta, const double dtmax){
		const double dta = aarseth_step(acc, jrk, snp, crk, d4a, d5a, eta);
		dt = dtmax;
		while(dt > dta) dt *= 0.5;
	}

	void correct(const Force &f, const double eta, const double etapow, const double dtlim){
#if 0
		const double h = 0.5 * dt;
		const double hinv = 2.0/dt;
#else
		const InvForDt tmp(dt);
		const double h    = tmp.h;
		const double hinv = tmp.hinv;
#endif

		const dvec3 Ap = (f.acc + acc);
		const dvec3 Am = (f.acc - acc);
		const dvec3 Jp = (f.jrk + jrk)*h;
		const dvec3 Jm = (f.jrk - jrk)*h;
		const dvec3 Sp = (f.snp + snp)*(h*h);
		const dvec3 Sm = (f.snp - snp)*(h*h);

		// do correct
		dvec3 vel1 = vel + h*(Ap - 0.4*Jm + (1./15)*Sp);
		pos += h*((vel + vel1) + h*(-0.4*Am + (1./15)*Jp));
		vel = vel1;
		acc = f.acc;
		jrk = f.jrk;
		snp = f.snp;
		tlast += dt;

		// taylor series
		double hinv2 = hinv*hinv; 
		double hinv3 = hinv2*hinv; 
		double hinv4 = hinv2*hinv2;
		double hinv5 = hinv2*hinv3;

		crk = (0.75*hinv3)*(-5.*Am + 5.*Jp - Sm);
		d4a = ( 1.5*hinv4)*(-Jm + Sp);
		d5a = ( 7.5*hinv5)*(3*Am - 3*Jp + Sm);

		crk += h*(d4a + 0.5*h*d5a);
		d4a += h*d5a;

		// update timestep
#if 0
		const double dta = aarseth_step(acc, jrk, snp, crk, d4a, d5a, eta);
		dt = dtlim;
		while(dt > dta) dt *= 0.5;
#else
		const double dtq = aarseth_step_quant(acc, jrk, snp, crk, d4a, d5a, etapow);
		dt = dtq<dtlim ? dtq : dtlim;
#endif
	}


#ifdef __AVX__
	static void copy_particle(Particle &dst, const Particle &src){
		const v4df *vsrc = (const v4df *)&src;
		const v4df ym0 = vsrc[0];
		const v4df ym1 = vsrc[1];
		const v4df ym2 = vsrc[2];
		const v4df ym3 = vsrc[3];
		const v4df ym4 = vsrc[4];
		const v4df ym5 = vsrc[5];
		const v4df ym6 = vsrc[6];

		v4df *vdst = (v4df *)&dst;
		vdst[0] = ym0;
		vdst[1] = ym1;
		vdst[2] = ym2;
		vdst[3] = ym3;
		vdst[4] = ym4;
		vdst[5] = ym5;
		vdst[6] = ym6;
	}
	Particle(const Particle &p){
		copy_particle(*this, p);
	}
	const Particle &operator=(const Particle &p){
		copy_particle(*this, p);
		return (*this);
	}
#endif
}
__attribute__((aligned(32)));

#ifdef HPC_ACE_GRAVITY
static inline void swap(Particle &a, Particle &b){
	asm("# fast SIMD swap");
	typedef __builtin_v2r8 v2r8;
	v2r8 *va = (v2r8 *)&a;
	v2r8 *vb = (v2r8 *)&b;
	const v2r8 va0 = va[0], vb0 = vb[0];
	const v2r8 va1 = va[1], vb1 = vb[1];
	const v2r8 va2 = va[2], vb2 = vb[2];
	const v2r8 va3 = va[3], vb3 = vb[3];
	const v2r8 va4 = va[4], vb4 = vb[4];
	const v2r8 va5 = va[5], vb5 = vb[5];
	const v2r8 va6 = va[6], vb6 = vb[6];
	const v2r8 va7 = va[7], vb7 = vb[7];
	const v2r8 va8 = va[8], vb8 = vb[8];
	const v2r8 va9 = va[9], vb9 = vb[9];
	const v2r8 va10 = va[10], vb10 = vb[10];
	const v2r8 va11 = va[11], vb11 = vb[11];
	const v2r8 va12 = va[12], vb12 = vb[12];
	const v2r8 va13 = va[13], vb13 = vb[13];
#define STORE(ptr, val) __builtin_fj_store_v2r8((double *)(ptr), val)
	STORE(va+0 , vb0 ), STORE(vb+0 , va0 );
	STORE(va+1 , vb1 ), STORE(vb+1 , va1 );
	STORE(va+2 , vb2 ), STORE(vb+2 , va2 );
	STORE(va+3 , vb3 ), STORE(vb+3 , va3 );
	STORE(va+4 , vb4 ), STORE(vb+4 , va4 );
	STORE(va+5 , vb5 ), STORE(vb+5 , va5 );
	STORE(va+6 , vb6 ), STORE(vb+6 , va6 );
	STORE(va+7 , vb7 ), STORE(vb+7 , va7 );
	STORE(va+8 , vb8 ), STORE(vb+8 , va8 );
	STORE(va+9 , vb9 ), STORE(vb+9 , va9 );
	STORE(va+10, vb10), STORE(vb+10, va10);
	STORE(va+11, vb11), STORE(vb+11, va11);
	STORE(va+12, vb12), STORE(vb+12, va12);
	STORE(va+13, vb13), STORE(vb+13, va13);
#undef STORE
}
#endif

#ifdef AVX_GRAVITY
#include "hermite6-avx.h"
#elif defined HPC_ACE_GRAVITY
#include "hermite6-k.h"
#elif defined MX_GRAVITY
#include "hermite6-mx.h"
#elif defined MIC_GRAVITY
#include "hermite6-mic.h"
#elif defined CUDA_TITAN
#include "hermite6-titan.h"
#else
struct Gravity{
	typedef Particle GParticle;
	struct GPredictor{
		dvec3  pos;
		double mass;
		dvec3  vel;
		long   id;
		dvec3  acc; // 11 DP

		GPredictor(const GParticle &p, const double tsys){
			const double dt = tsys - p.tlast;
			const double dt2 = (1./2.) * dt;
			const double dt3 = (1./3.) * dt;
			const double dt4 = (1./4.) * dt;
			const double dt5 = (1./5.) * dt;

			pos  = p.pos + dt * (p.vel + dt2 * (p.acc + dt3 * (p.jrk + dt4 * (p.snp + dt5 * (p.crk)))));
			vel  = p.vel + dt * (p.acc + dt2 * (p.jrk + dt3 * (p.snp + dt4 * (p.crk))));
			acc  = p.acc + dt * (p.jrk + dt2 * (p.snp + dt3 * (p.crk)));
			mass = p.mass;
			id   = p.id;
		}
	};

	const int  nbody;
	GParticle  *ptcl;
	GPredictor *pred;


	Gravity(const int _nbody) : nbody(_nbody) {
		ptcl = allocate<GParticle,  64> (nbody);
		pred = allocate<GPredictor, 64> (nbody);
	}
	~Gravity(){
		deallocate<GParticle,  64> (ptcl);
		deallocate<GPredictor, 64> (pred);
	}

	void set_jp(const int addr, const Particle &p){
		ptcl[addr] = p;
	}

	void predict_all(const double tsys){
#pragma omp parallel for
		for(int i=0; i<nbody; i++){
			pred[i] = GPredictor(ptcl[i], tsys);
		}
	}

	void calc_force_on_first_nact(
			const int    nact,
			const double eps2,
			Force        force[] )
	{
		const int ni = nact;
		const int nj = nbody;
#pragma omp parallel for
		for(int i=0; i<ni; i++){
			const dvec3 posi = pred[i].pos;
			const dvec3 veli = pred[i].vel;
			const dvec3 acci = pred[i].acc;
			dvec3 acc(0.0);
			dvec3 jrk(0.0);
			dvec3 snp(0.0);
			for(int j=0; j<nj; j++){
				const dvec3  dr    = pred[j].pos - posi;
				const dvec3  dv    = pred[j].vel - veli;
				const dvec3  da    = pred[j].acc - acci;

				const double r2    = eps2 + dr*dr;
				const double drdv  = dr*dv;
				const double dvdv  = dv*dv;
				const double drda  = dr*da;

				const double ri2   = 1.0 / r2;
				const double mri3  = pred[j].mass * ri2 * sqrt(ri2);
				const double alpha = drdv * ri2;
				const double beta  = (dvdv + drda) * ri2 + alpha * alpha;
				
#if 0
				const dvec3 aij = mri3 * dr;
				const dvec3 jij = mri3 * dv + (-3.0*alpha) * aij;
				const dvec3 sij = mri3 * da + (-6.0*alpha) * jij + (-3.0*beta) * aij;

				acc += aij;
				jrk += jij;
				snp += sij;
#else
				acc += mri3 * dr;
				dvec3 tmp = dv + (-3.0*alpha) * dr;
				jrk += mri3 * tmp;
				snp += mri3 * (da + (-6.0*alpha) * tmp + (-3.0*beta) * dr);
#endif
			}
			force[i].acc = acc;
			force[i].jrk = jrk;
			force[i].snp = snp;
		}
	}

	void calc_potential(
			const double eps2,
			double       potbuf[] )
	{
		const int ni = nbody;
		const int nj = nbody;
#pragma omp parallel for
		for(int i=0; i<ni; i++){
			double pot = 0.0;
			const dvec3 posi = ptcl[i].pos;
			for(int j=0; j<nj; j++){
				// if(j == i) continue;
				const dvec3  posj = ptcl[j].pos;
				const dvec3  dr   = posj - posi;
				const double r2   = eps2 + dr*dr;
				const double mj = (j != i) ? ptcl[j].mass : 0.0;
				pot -= mj / sqrt(r2);
			}
			potbuf[i] = pot;
		}
	}
};
#endif // !defined AVX_GRAVITY
