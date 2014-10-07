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
		double3 snp;
		double3 crk; // double * 20
	};

	struct GPredictor{
		double3 pos;
		double  mass;
		double3 vel;
		double3 acc; // double * 10
	};

	struct GForce{
		double3 acc;
		double3 jrk;
		double3 snp; // double * 9
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

	void set_jp(const int addr, const Particle p){
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

		pdst.snp.x = p.snp.x;
		pdst.snp.y = p.snp.y;
		pdst.snp.z = p.snp.z;

		pdst.crk.x = p.crk.x;
		pdst.crk.y = p.crk.y;
		pdst.crk.z = p.crk.z;
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
