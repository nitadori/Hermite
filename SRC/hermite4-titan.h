#pragma once
#include "cuda_pointer.h"

struct Gravity{
	enum{
		// NJBLOCK  = 28, // for Titan
		// NJREDUCE = 32,
		NJBLOCK  = 56, // for Titan
		NJREDUCE = 64,
		NTHREAD  = 64,
		NIMAX    = 2048,
	};

	struct GParticle{
		double3 pos;
		double  mass;
		double3 vel;
		double  tlast;
		double3 acc;
		double3 jrk;
	};

	struct GPredictor{
		double3 pos;
		double  mass;
		double3 vel;
	};

	struct GForce{
		double3 acc;
		double3 jrk;
	};

	const int  nbody;

	cudaPointer<GParticle > ptcl;
	cudaPointer<GPredictor> pred;

	cudaPointer <GForce[NJBLOCK]> fpart;
	cudaPointer <GForce         > ftot ;

	int njpsend;


	Gravity(const int _nbody) : nbody(_nbody) {
		ptcl .allocate(nbody + NTHREAD);
		pred .allocate(nbody + NTHREAD);
		fpart.allocate(NIMAX);
		ftot .allocate(NIMAX);

		njpsend = nbody;
	}
	~Gravity(){
		ptcl .free();
		pred .free();
		fpart.free();
		ftot .free();
	}

	static double3 make_double3(const dvec3 &v){
		return ::make_double3(v.x, v.y, v.z);
	}

	void set_jp(const int addr, const Particle p){
		GParticle &pdst = ptcl[addr];
		pdst.pos   = make_double3(p.pos);
		pdst.mass  = p.mass;
		pdst.vel   = make_double3(p.vel);
		pdst.tlast = p.tlast;
		pdst.acc   = make_double3(p.acc);
		pdst.jrk   = make_double3(p.jrk);
	}

	void predict_all(const double tsys);

	void calc_force_on_first_nact(
			const int    nact,
			const double eps2,
			Force        force[] )
	{
		for(int ii=0; ii<nact; ii+=NIMAX){
			const int ni = (nact-ii) < NIMAX ? (nact-ii) : NIMAX;
			calc_force_in_range(ii, ii+ni, eps2, force);
		}

		njpsend = nact;

		// printf("computed nact=%d\n", nact);

		// puts("init force done");
		// exit(0);
	}

	void calc_potential(
			const double eps2,
			double       potbuf[] );

private:
	void calc_force_in_range(
			const int    is,
			const int    ie,
			const double eps2,
			Force        force[] );
};
