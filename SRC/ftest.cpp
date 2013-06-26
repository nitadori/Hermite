#include <cstdio>
#include <cassert>
#include "vector3.h"

#ifdef __SSE__
#warning SSE is available
typedef double v2df __attribute__((vector_size(16)));
#endif
#ifdef __AVX__
#warning AVX is available
typedef double v4df __attribute__((vector_size(32)));
#endif
#ifdef __HPC_ACE__
#warning HPC-ACE is available
#endif

#include "allocate.h"
#include "hermite4.h"
#include "nbodysystem.h"
int main(){
	NbodySystem sys;
	sys.read_snapshot_berczic("data.inp");
	const double eta   = 0.1;
	const double eta_s = 0.01;
	const double tend  = 1.0;
	sys.set_run_params(tend, 1./16., 0.0, eta, eta_s);
	sys.set_thermal_eps(4.0);

	sys.prof.flush();
	sys.prof.beg(Profile::TOTAL);
	sys.init_step();

	int nbody = sys.nbody;
	sys.sort_ptcl_by_id(nbody);

	sys.prof.end(Profile::TOTAL);
	sys.prof.show();
	for(int i=0; i<nbody; i++){
		const Particle &p = sys.ptcl[i];
		printf("%4d %A %A %A  %a\n", p.id, p.acc.x, p.acc.y, p.acc.z, sys.potbuf[i]);
	}
	return 0;
}
