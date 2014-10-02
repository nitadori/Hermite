#pragma once
#include "cuda_pointer.h"

struct Gravity{
	enum{
		NJBLOCK  = 30, // for Titan
		NJREDUCE = 32,
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
		ptcl .allocate(nbody);
		pred .allocate(nbody);
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

	void set_jp(const int addr, const Particle &p){
		GParticle &pdst = ptcl[addr];
		pdst.pos.x = p.pos.x;
		pdst.pos.y = p.pos.y;
		pdst.pos.z = p.pos.z;
		pdst.mass  = p.mass;

		pdst.vel.x = p.vel.x;
		pdst.vel.y = p.vel.y;
		pdst.vel.z = p.vel.z;
		pdst.tlast = p.tlast;

		pdst.acc.x = p.acc.x;
		pdst.acc.y = p.acc.y;
		pdst.acc.z = p.acc.z;

		pdst.jrk.x = p.jrk.x;
		pdst.jrk.y = p.jrk.y;
		pdst.jrk.z = p.jrk.z;
	}

	void predict_all(const double tsys);

	void calc_force_on_first_nact(
			const int    nact,
			const double eps2,
			Force        force[] )
	{
		njpsend = nact;
	}

	void calc_potential(
			const double eps2,
			double       potbuf[] );
};
