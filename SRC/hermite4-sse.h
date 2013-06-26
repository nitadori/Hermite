#include <omp.h>


struct Gravity{
	enum{
		NIMAX = 1024,
		MAXTHREAD = 64,
	};

	typedef double v2df __attribute__((vector_size(16)));

	struct GParticle{
		v2df tlast;
		v2df mass;
		v2df pos[3];
		v2df vel[3];
		v2df acc[3];
		v2df jrk[3];
	};

	struct GPredictor{
		v2df mass;
		v2df pos[3];
		v2df vel[3];
	};

	const int  nbody;
	GParticle  *ptcl;
	GPredictor *pred;

	Gravity(const int _nbody) : nbody(_nbody) {
		ptcl = allocate<GParticle,  64> (1 + nbody/2);
		pred = allocate<GPredictor, 64> (1 + nbody/2);
#pragma omp parallel
		assert(MAXTHREAD >= omp_get_thread_num());
	}

	~Gravity(){
		deallocate<GParticle,  64> (ptcl);
		deallocate<GPredictor, 64> (pred);
	}

	void set_jp(const int addr, const Particle & _restrict p){
		const int ah = addr/2;
		const int al = addr%2;
		double *ptr = (double *)(&ptcl[ah]) + al;
		ptr[ 0] = p.tlast;
		ptr[ 2] = p.mass;
		ptr[ 4] = p.pos.x;
		ptr[ 6] = p.pos.y;
		ptr[ 8] = p.pos.z;
		ptr[10] = p.vel.x;
		ptr[12] = p.vel.y;
		ptr[14] = p.vel.z;
		ptr[16] = p.acc.x;
		ptr[18] = p.acc.y;
		ptr[20] = p.acc.z;
		ptr[22] = p.jrk.x;
		ptr[24] = p.jrk.y;
		ptr[26] = p.jrk.z;
	}

	void predict_all(const double tsys){
		const v2df tnow = {tsys, tsys};
#pragma omp parallel for
		for(int i=0; i<nbody; i+=2){
			const GParticle &p = ptcl[i];
			const v2df dt  = tnow - p.tlast;
			const v2df dt2 = dt * v2df{1./2., 1./2.};
			const v2df dt3 = dt * v2df{1./3., 1./3.};

			const v2df pos = p.pos + dt * (p.vel + dt2 * (p.acc + dt3 * (p.jrk)));
			const v2df vel = p.vel + dt * (p.acc + dt2 * (p.jrk));
			
			pred[i].mass = p.mass;
			pred[i].pos  = pos;
			pred[i].vel  = vel;
		}
	}

	void calc_force_in_range(
			const int    is,
			const int    ie,
			const double deps2,
			Force        force[] )
	{
#pragma omp parallel
		{
			for(int i=is; i<ie; i++){
#pragma omp for // calculate partial force
				for(int j=0; j<nj; j+=2){
				} // for(j)
			}
		} // end omp parallel
	}

	void calc_force_on_first_nact(
			const int    nact,
			const double eps2,
			Force        force[] )
	{
		for(int ii=0; ii<nact; ii+=NIMAX){
			const int ni = (nact-ii) < NIMAX ? (nact-ii) : NIMAX;
			calc_force_in_range(ii, ii+ni, eps2, force);
		}
	}

	void calc_potential(
			const double deps2,
			double       potbuf[] )
	{
#pragma omp parallel for
		for(int i=0; i<ni; i++){
			for(int j=0; j<nj; j+=2){
			}
		}
	}
};
