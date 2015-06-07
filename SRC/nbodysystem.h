#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <sys/time.h>
#include <unistd.h>

#include <omp.h>
// #include <mpi.h>

#include "vector3.h"

// #define NOPROFILE

#ifdef NOPROFILE
struct Profile{
	enum{
		FORCE = 0,
		POT,
		PREDICT,
		CORRECT,
		SORT,
		SCAN,
		SET_JP,
		IO,
		// don't touch the below
		TOTAL,
		MISC,
		NUM_ELEM,
	};
	void flush(){}
	void beg(const int elem, const bool reuse = false){}
	void end(const int elem){}
	void beg_master(const int elem, const bool reuse = false){}
	void end_master(const int elem){}
	void show(
			FILE *fp = stderr,
			const char *fmt = " %s : %e\n")
	{}

	static double wtime(){
#if 1
		return 0.0;
#else
# if 0
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return double(tv.tv_sec) + 1.e-6*double(tv.tv_usec);
# else
		return omp_get_wtime();
# endif
#endif
	}
};
#else
struct Profile{
	enum{
		FORCE = 0,
		POT,
		PREDICT,
		CORRECT,
		SORT,
		SCAN,
		SET_JP,
		IO,
		// don't touch the below
		TOTAL,
		MISC,
		NUM_ELEM,
	};

	static const char *name(const int i){
		static const char *strs[NUM_ELEM] = {
			"force   ",
			"pot     ",
			"predict ",
			"correct ",
			"sort    ",
			"scan    ",
			"set_jp  ",
			"I/O     ",
			"total   ",
			"misc    ",
		};
		return strs[i];
	}

	double time[NUM_ELEM];
	double tprev;

	long num_step_prev;
	long num_bstep_prev;
	
	void flush(){
		for(int i=0; i<NUM_ELEM; i++) time[i] = 0.0;
	}
	void beg(const int elem, const bool reuse = false){
		if(reuse) time[elem] -= tprev;
		else      time[elem] -= (tprev=wtime());
	}
	void end(const int elem){
		// time[elem] += wtime();
		tprev = wtime();
		time[elem] += tprev;
	}
	void beg_master(const int elem, const bool reuse = false){
#pragma omp master
		{
			if(reuse) time[elem] -= tprev;
			else      time[elem] -= (tprev=wtime());
		}
	}
	void end_master(const int elem){
#pragma omp master
		{
			// time[elem] += wtime();
			tprev = wtime();
			time[elem] += tprev;
		}
	}
	void show(
			const long nbody,
			const long num_step_tot,
			const long num_bstep_tot,
			FILE *fp = stderr,
			const char *fmt = " %s : %e sec, %e usec : %6.2f %%\n")
	{
		const long ns  = num_step_tot  - num_step_prev;
		const long nbs = num_bstep_tot - num_bstep_prev;
		num_step_prev  = num_step_tot;
		num_bstep_prev = num_bstep_tot;

		time[MISC] = time[TOTAL];
		for(int i=0; i<NUM_ELEM-2; i++){
			time[MISC] -= time[i];
		}
		for(int i=0; i<NUM_ELEM; i++){
			fprintf(fp, fmt, name(i), 
					time[i], 
					time[i] * (1.e6/nbs),
					100.0 * time[i] / time[TOTAL]);
		}

		const double nact   = double(ns) / double(nbs);
		const double wtime  = time[TOTAL];
		const double Gflops = (((1.0e-9 * ns) * nbody) * Particle::flops) / wtime;
		fprintf(stdout, "## nbody  wtime(sec)  Gflops  wtime/block(usec)  nact\n");
		fprintf(stdout, "## %ld %e %e %e %e\n",
				nbody, wtime, Gflops, wtime/nbs*1.e6, nact);
#if 1
		size_t nread  = nbs * (nbody * sizeof(Gravity::GParticle));
		size_t nwrite = nbs * (nbody * sizeof(Gravity::GPredictor));
		double bandwidth = (nread + nwrite) / time[PREDICT];
		fprintf(stdout, "prediction bandwidth = %f GB/s\n",
				bandwidth * 1.e-9);
#endif
		fflush(fp);
		fflush(stdout);
	}

	static double wtime(){
#ifdef __HPC_ACE__
#  if 0
		return 1.e-6 * __gettod();
#  else
		static bool initcall = true;
		static double inv_cpu_clock = 0.0;
		if(initcall){
			initcall = false;
			unsigned long x0, x1;
			double t0 = omp_get_wtime();
			asm volatile ("rd %%tick, %0" : "=r" (x0));
			sleep(1);
			asm volatile ("rd %%tick, %0" : "=r" (x1));
			double t1 = omp_get_wtime();
			inv_cpu_clock = (t1-t0) / (x1-x0);
			printf("HPC_ACE at %e Hz\n", 1.0 / inv_cpu_clock);
		}
		unsigned long x;
		asm volatile ("rd %%tick, %0" : "=r" (x));
		return (1.0e-6 / 1650) * (double)x;
#  endif
#else
# if 0
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return double(tv.tv_sec) + 1.e-6*double(tv.tv_usec);
# else
		return omp_get_wtime();
# endif
#endif
	}
};
#endif

struct RAII_Timer{
	const char *name, *fmt;
	FILE *fp;
	double tbeg, tend;
	RAII_Timer(
			const char *_name = "timer",
			FILE       *_fp   = stdout,
			const char *_fmt  = "%s: %e sec\n")
		: name(_name), fmt(_fmt), fp(_fp)
	{
		tbeg = Profile::wtime();
	}
	~RAII_Timer(){
		tend = Profile::wtime();
		fprintf(fp, fmt, name, tend-tbeg);
	}
};

#if 0
template <typename T, size_t align>
T *allocate(size_t num_elem){
	void *ptr;
	int ret = posix_memalign(&ptr, align, num_elem * sizeof(T));
	assert(0 == ret);
	assert(ptr);
	return (T *)ptr;
}
#endif

struct CmpPtcl_dt{
	bool operator()(const Particle &p1, const Particle &p2) const {
		return (p1.dt < p2.dt);
	}
};
struct CmpPtcl_id{
	bool operator()(const Particle &p1, const Particle &p2) const {
		return (p1.id < p2.id);
	}
};

struct Split_dt{
#if 1
	const double dt;
	Split_dt(const double _dt) : dt(_dt) {}
	bool operator()(const Particle &p){
		return (p.dt < dt);
	}
#else
	const long dtl;
	Split_dt(const double _dt) : dtl(*(long *)&_dt) {}
	bool operator()(const Particle &p){
		return (*(long *)&p.dt < dtl);
	}
#endif
};

struct NbodySystem{
	// paramters
	long   nbody;
	double eps2;
	double eta, etapow, eta_s;
	double tsys, tend;
	double dtmax;
	// counters
	double init_energy, prev_energy;
	long   num_step, num_bstep;
	long   num_step_tot, num_bstep_tot;
	// buffers
	Particle *ptcl;
	Force    *force;
	double   *potbuf;
	double   *dtbuf;
	// int      *ilist;
	// objects
	Gravity  *gravity;
	Profile   prof;


	NbodySystem(){
		nbody = 0;
		num_step      = 0;
		num_bstep     = 0;
		num_step_tot  = 0;
		num_bstep_tot = 0;
		dtmax = 0.0;
		eta = etapow = eta_s = 0.0;
		ptcl   = NULL;
		force  = NULL;
		potbuf = NULL;
		dtbuf  = NULL;
		// ilist = NULL;
		// schedule = NULL;
		gravity  = NULL;
	}
	~NbodySystem(){
		release_resources();
	}

	void allocate_resources(){
		fprintf(stderr, "allocate resources for n = %ld\n", nbody);
		assert(nbody > 0);

		release_resources();

		ptcl   = allocate<Particle, 64> (nbody);
		force  = allocate<Force,    64> (nbody);
		potbuf = allocate<double,   64> (nbody);
		dtbuf  = allocate<double,   64> (nbody);
		// ilist = allocate<int,      64> (nbody);

		// schedule = new Schedule(dtmax);
		gravity  = new Gravity(nbody);

		prof.num_step_prev  = num_step_tot;
		prof.num_bstep_prev = num_bstep_tot;
#pragma omp parallel
		{
#pragma omp master
			fprintf(stderr, "NUM_THREADS = %d\n", omp_get_num_threads());
		}
	}
	void release_resources(){
		deallocate<Particle, 64>(ptcl  ); ptcl   = NULL;
		deallocate<Force   , 64>(force ); force  = NULL;
		deallocate<double  , 64>(potbuf); potbuf = NULL;
		deallocate<double  , 64>(dtbuf ); dtbuf  = NULL;
		// free(ilist); ilist = NULL;

		// delete schedule; schedule = NULL;
		delete gravity;  gravity  = NULL;
	}
#if 0
	void read_snapshot_berczic(const char *filename){
		FILE *fp = fopen(filename, "r");
		assert(fp);
		int snapid;
		int nread = fscanf(fp, "%d %ld %lf", &snapid, &nbody,  &tsys);
		assert(3 == nread);
		fprintf(stderr, "read snapshot, n = %ld\n", nbody);

		allocate_resources();

		for(int i=0; i<nbody; i++){
			int id;
			double mass;
			dvec3 pos, vel;
			nread = fscanf(fp, "%d %lf %lf %lf %lf %lf %lf %lf",
					&id, &mass, &pos.x, &pos.y, &pos.z, &vel.x, &vel.y, &vel.z);
			assert(8 == nread);
			// The constructer flushes higher derivatives
			ptcl[i] = Particle(id, mass, pos, vel);
#ifdef EIGHTHDD
			const dvec3 off(sqrt(3.0) * pow(2.0, 32.0));
			ptcl[i].posL = off;
			dd_twosum(ptcl[i].posH, ptcl[i].posL);
#endif
		}

		fclose(fp);
		fprintf(stderr, "read snapshot, done\n");
	}
#endif
	void write_snapshot_masaki(const char *filename, const int snapid){
		FILE *fp = fopen(filename, "w");
		assert(fp);

		fprintf(fp, "%d\n%ld\n%f\n", snapid, nbody, tsys);
		for(int i=0; i<nbody; i++){
			const Particle &p = ptcl[i];
#ifndef EIGTHDD
			const dvec3 pos = p.pos;
#else
			const dvec3 pos = p.posH + p.posL;
#endif
			fprintf(fp, "%6ld    %A    %A %A %A    %A %A %A    %A\n",
					p.id, p.mass, 
					pos.x, pos.y, pos.z,
					p.vel.x, p.vel.y, p.vel.z,
					potbuf[i]);
		}
		fclose(fp);
	}
	void read_snapshot_masaki(const char *filename, int &snapid){
		int nread;
		FILE *fp = fopen(filename, "r");
		assert(fp);

		nread = fscanf(fp, "%d %ld %lf", &snapid, &nbody, &tsys);
		assert(3 == nread);
		fprintf(stderr, "read snapshot (form M.I.), n = %ld, t = %f\n", nbody, tsys);

		allocate_resources();

		for(int i=0; i<nbody; i++){
			long id;
			double mass, pot;
			dvec3 pos, vel;
			nread = fscanf(fp, "%ld %lA %lA %lA %lA %lA %lA %lA %lA", 
					&id, &mass,
					&pos.x, &pos.y, &pos.z,
					&vel.x, &vel.y, &vel.z,
					&pot);
			assert(9 == nread);
			ptcl[i] = Particle(id, mass, pos, vel);
		}
		fclose(fp);
	}

	void dump(const char *filename, const int dumpid){
		FILE *fp = fopen(filename, "w");
		assert(fp);

#if 0
		sort_ptcl_by_id(nbody);
#endif

		fprintf(fp, "%d %d\n%ld\n%20.16f\n", Particle::order, dumpid, nbody, tsys);
		fprintf(fp, "%A %A\n", init_energy, prev_energy);
		fprintf(fp, "%ld %ld\n", num_step_tot, num_bstep_tot);
		for(int i=0; i<nbody; i++){
			ptcl[i].dump(fp);
		}
		fclose(fp);

#if 0
		sort_ptcl(nbody, dtmax);
		for(int i=0; i<nbody; i++){
			gravity->set_jp(i, ptcl[i]);
		}
#endif
	}
	void restore(const char *filename, int &dumpid){
		FILE *fp = fopen(filename, "r");
		assert(fp);
		int order;
		assert(4 == fscanf(fp, "%d %d %ld %lf", &order, &dumpid, &nbody, &tsys));
		assert(2 == fscanf(fp, "%lA %lA", &init_energy, &prev_energy));
		assert(2 == fscanf(fp, "%ld %ld", &num_step_tot, &num_bstep_tot));
		assert(order == Particle::order);
		fprintf(stderr, "read dump file : <%s>, order=%d, nbody=%ld, tsys=%f\n", 
				filename, order, nbody, tsys);

		allocate_resources();

		for(int i=0; i<nbody; i++){
			ptcl[i].restore(fp);
			ptcl[i].tlast = tsys;
		}
		fclose(fp);
	}
	void reinitialize_steps(const double _eta, const double _dtmax){
		eta   = _eta;
		dtmax = _dtmax;
		for(int i=0; i<nbody; i++){
			ptcl[i].recalc_dt(eta, dtmax);
		}

		sort_ptcl(nbody, dtmax);
		for(int i=0; i<nbody; i++){
			gravity->set_jp(i, ptcl[i]);
		}
	}

	void set_run_params(
			const double _tend,
			const double _dtmax,
			const double _eps2,
			const double _eta,
			const double _eta_s)
	{
		tend  = _tend;
		dtmax = _dtmax;
		eps2  = _eps2;
		eta   = _eta;
		eta_s = _eta_s;

		const int ipow = 2*(Particle::order - 3);
		etapow = pow(eta, double(ipow));

		// if(schedule) delete schedule;
		assert(dtmax > 0.0);
		// schedule = new Schedule(dtmax);
	}
	void set_thermal_eps(const double kt){
		const double eps = 4.0 / nbody / kt;
		fprintf(stderr, "eps is set to %e\n", eps);
		eps2 = eps * eps;
	}

	__attribute__((noinline))
	void sort_ptcl(const int nact, const double dtlim){
#if 0
		sort_ptcl(nact);
#else
		// this version is faster than the version with std::sort
		Particle *beg = ptcl;
		Particle *end = ptcl+nact;
		double dt = dtlim;
		while(end != beg){
			end = std::partition(beg, end, Split_dt(dt));
			dt *= 0.5;
		}
#endif
		// for next count_nact
		for(int i=0; i<nact; i++){
			dtbuf[i] = ptcl[i].dt + ptcl[i].tlast;
		}
	}
	__attribute__((noinline))
	void sort_ptcl_dtcache(const int nact, const double dtlim, const bool do_set_jp = false){
		int beg = 0;
		int end = nact;
		union{ double d; long l; } m64;
		m64.d = dtlim;
		long dtl = m64.l;
		const long *dtls = (const long *)dtbuf;
		while(end != beg){
			int i = beg;
			int j = end;
			while(true){
				while(true){
					if(i == j)        goto breakpoint;
					if(dtls[i] != dtl) ++i;
					else              break;
				}
				--j;
				while(true){
					if(i == j)          goto breakpoint;
					if(dtls[j] == dtl)  --j;
					else                break;
				}
				std::swap(dtbuf[i], dtbuf[j]);
#ifdef MIC_GRAVITY
				const VParticle pi(&ptcl[i]);
				const VParticle pj(&ptcl[j]);
				pi.store(&ptcl[j]);
				pj.store(&ptcl[i]);
#elif defined HPC_ACE_GRAVITY
				swap(ptcl[i], ptcl[j]);
#else
				std::swap(ptcl[i], ptcl[j]);
#endif
				if(do_set_jp){
					gravity->set_jp(i, ptcl[i]);
					gravity->set_jp(j, ptcl[j]);
				}

				++i;
			}
breakpoint:
			end = i;
			// printf("done : %lx\n", dtl);
			dtl -= 1L << 52;
		}
		// for next count_nact
		for(int i=0; i<nact; i++){
			dtbuf[i] = ptcl[i].dt + ptcl[i].tlast;
		}
	}

	void sort_ptcl(const int nact){
		std::sort(ptcl+0, ptcl+nact, CmpPtcl_dt());
	}

	void sort_ptcl_by_id(const int nact){
		std::sort(ptcl+0, ptcl+nact, CmpPtcl_id());
	}

	void init_step(){
		const int n = nbody;
		for(int i=0; i<n; i++){
			ptcl[i].tlast = tsys;
			gravity->set_jp(i, ptcl[i]);
			// ilist[i] = i;
		}
		init_energy = prev_energy = calc_energy(true);
		if(Particle::order > 4){ 
			puts("INITIAL FORCE");
			gravity->predict_all(tsys);
			gravity->calc_force_on_first_nact(nbody, eps2, force);
			for(int i=0; i<n; i++){
				ptcl[i].assign_force(force[i]);
				gravity->set_jp(i, ptcl[i]);
			}
		}
		prof.beg(Profile::PREDICT);
		gravity->predict_all(tsys);
		prof.end(Profile::PREDICT);

		prof.beg(Profile::FORCE);
		const double t0 = prof.tprev;
		gravity->calc_force_on_first_nact(nbody, eps2, force);
		prof.end(Profile::FORCE);
		const double t1 = prof.tprev;
		{
			const double Gflops = (((1.0e-9 * nbody) * nbody) * Particle::flops) / (t1-t0);
			printf("## Init N square : %f Gflops\n", Gflops);
		}
#if 0
		for(int i=n-10; i<n; i++){
			dvec3 acc = force[i].acc;
			dvec3 jrk = force[i].jrk;
			printf("%d : (%e, %e, %e) (%e, %e, %e)\n", 
					i, 
					acc.x, acc.y, acc.z,
					jrk.x, jrk.y, jrk.z);
		}
		exit(0);
#endif
		for(int i=0; i<n; i++){
			ptcl[i].assign_force(force[i]);
			ptcl[i].init_dt(eta_s, dtmax);
#if 0
				dvec3 acc = ptcl[i].jrk;
				printf("%d : (%e, %e, %e)\n", i, acc.x, acc.y, acc.z);
#endif
		}
		// sort_ptcl(nbody);
		sort_ptcl(nbody, dtmax);
		for(int i=0; i<n; i++){
#if 0
			printf("%4ld, %A, %f\n", 
					ptcl[i].id, ptcl[i].dt, ptcl[i].acc.abs()/ptcl[i].jrk.abs());
#endif
#if 0
			printf("%4ld, %e, (%24.16e, %24.16e, %24.16e, %24.16e)\n", 
					ptcl[i].id, ptcl[i].dt, 
					ptcl[i].acc.abs(), ptcl[i].jrk.abs(),
					ptcl[i].snp.abs(), ptcl[i].crk.abs());
#endif
			gravity->set_jp(i, ptcl[i]);
		}
		init_energy = calc_energy(true);
		// exit(0);
	}

	double calc_energy(bool print=false){
		prof.beg(Profile::POT);
		gravity->calc_potential(eps2, potbuf);
		prof.end(Profile::POT);
		double ke = 0.0;
		double pe = 0.0;
		for(int i=0; i<nbody; i++){
			ke += ptcl[i].mass * ptcl[i].vel.norm2();
			pe += ptcl[i].mass * potbuf[i];
		}
		ke *= 0.5;
		pe *= 0.5;
		if(print){
			fprintf(stderr, "ke = %24.16e, pe = %24.16e, e = %24.16e\n",
					ke, pe, ke+pe);
		}
		return ke+pe;
	}

	double calc_dtlim(const double tnext) const {
		double dtlim = dtmax;
		double s = tnext / dtmax;
		while(s != double(int(s))){
			s *= 2.0;
			dtlim *= 0.5;
			assert(dtlim >= 1.0/(1LL<<32));
		}
		return dtlim;
	}

	int count_nact(const double tnext) const {
#if 1
		int nact;
		for(nact = 0; nact<nbody; nact++){
#if 0
			const Particle &p = ptcl[nact];
			if(p.dt + p.tlast != tnext) break;
#else
			if(dtbuf[nact] != tnext) break;
#endif
		}
		return nact;
#else
		int nact = 0;
		int skip = 64;
		for(;;){
			if(nact == nbody) return nact;
			if(nact < nbody && dtbuf[nact+skip] == tnext){
				nact += skip;
				continue;
			}else{
				if(1 == skip) return nact+1;
				skip /= 4;
				continue;
			}
		}
#endif
	}

	__attribute__((noinline))
	void integrate_one_block(){
		prof.beg(Profile::SCAN, true);
		const double tnext = ptcl[0].tlast + ptcl[0].dt;
		const double dtlim = calc_dtlim(tnext);
		const int    nact  = count_nact(tnext);
		prof.end(Profile::SCAN);
#if 0
		printf("t = %f, nact = %6d, dtlim = %A\n", tsys, nact, dtlim);
#endif
	  prof.beg(Profile::PREDICT);
		gravity->predict_all(tnext);
	  prof.end(Profile::PREDICT);

	  prof.beg(Profile::FORCE, true);
		gravity->calc_force_on_first_nact(nact, eps2, force);
	  prof.end(Profile::FORCE);

#if 0
	  prof.beg(Profile::CORRECT);
#pragma omp parallel for
		for(int i=0; i<nact; i++){
			ptcl[i].correct(force[i], eta, etapow, dtlim);
			dtbuf[i] = ptcl[i].dt;
		}
	  prof.end(Profile::CORRECT);

	  prof.beg(Profile::SORT, true);
		// sort_ptcl(nact);
#if 0
		sort_ptcl(nact, dtlim);
#else
		sort_ptcl_dtcache(nact, dtlim);
#endif
	  prof.end(Profile::SORT);

	  prof.beg(Profile::SET_JP, true);
#pragma omp parallel for
		for(int i=0; i<nact; i++){
			gravity->set_jp(i, ptcl[i]);
		}
	  prof.end(Profile::SET_JP);
#else
#pragma omp parallel
	  {
#pragma omp master
		  prof.beg(Profile::CORRECT);
#pragma omp for
		  for(int i=0; i<nact; i++){
			  ptcl[i].correct(force[i], eta, etapow, dtlim);
			  dtbuf[i] = ptcl[i].dt;
		  }
#pragma omp master
		  {
			  prof.end(Profile::CORRECT);
			  prof.beg(Profile::SORT, true);
			  sort_ptcl_dtcache(nact, dtlim);
			  prof.end(Profile::SORT);
			  prof.beg(Profile::SET_JP, true);
		  }
#pragma omp barrier
#pragma omp for nowait
		  for(int i=0; i<nact; i++){
			  gravity->set_jp(i, ptcl[i]);
		  }
	  } // end omp parallel
	  prof.end(Profile::SET_JP);
#endif

		num_step += nact;
		num_bstep++;

		tsys = tnext;
	}

#ifdef FAST_OMP_SYNC
	void integrate_one_dtmax_fast_omp(){
		const double tt = tsys + dtmax;
		double sh_tnext = dtbuf[0];
		double sh_dtlim = calc_dtlim(sh_tnext);
		int    sh_nact  = count_nact(sh_tnext);
#     pragma omp parallel
		{
			double tsys_loc = this->tsys;
			while(tsys_loc < tt){
				prof.beg_master(Profile::SCAN, true);
				// const double tnext = ptcl[0].tlast + ptcl[0].dt;
#if 0
				const double tnext = dtbuf[0];
				const double dtlim = calc_dtlim(tnext);
				const int    nact  = count_nact(tnext);
#else
				const double tnext = sh_tnext;
				const double dtlim = sh_dtlim;
				const int    nact  = sh_nact;
#endif
				prof.end_master(Profile::SCAN);

				prof.beg_master(Profile::PREDICT, true);
// #             pragma omp barrier
				gravity->predict_all_fast_omp(tnext);
				prof.end_master(Profile::PREDICT);
				prof.beg_master(Profile::FORCE, true);
#             pragma omp barrier
				gravity->calc_force_on_first_nact_fast_omp(nact, eps2, force);
				prof.end_master(Profile::FORCE);
// #             pragma omp barrier
#if 0
				prof.beg_master(Profile::CORRECT, true);
				if(nact > Gravity::NACT_PARALLEL_THRESH){
#                 pragma omp for
					for(int i=0; i<nact; i++){
						ptcl[i].correct(force[i], eta, etapow, dtlim);
						dtbuf[i] = ptcl[i].dt;
					}
				}else{
#                 pragma omp master
					for(int i=0; i<nact; i++){
						ptcl[i].correct(force[i], eta, etapow, dtlim);
						dtbuf[i] = ptcl[i].dt;
					}
				}
				prof.end_master(Profile::CORRECT);
				prof.beg_master(Profile::SORT, true);
#             pragma omp master
				{
					sort_ptcl_dtcache(nact, dtlim);
					this->num_step += nact;
					this->num_bstep++;
				}
#             pragma omp barrier
				prof.end_master(Profile::SORT);
				prof.beg_master(Profile::SET_JP, true);
				if(nact > Gravity::NACT_PARALLEL_THRESH){
#                 pragma omp for nowait
					for(int i=0; i<nact; i++){
						gravity->set_jp(i, ptcl[i]);
					}
				}else{
#                 pragma omp master
					for(int i=0; i<nact; i++){
						gravity->set_jp(i, ptcl[i]);
					}
				}
				prof.end_master(Profile::SET_JP);
#else
				if(nact > Gravity::NACT_PARALLEL_THRESH){
					prof.beg_master(Profile::CORRECT, true);
#                 pragma omp for
					for(int i=0; i<nact; i++){
						ptcl[i].correct(force[i], eta, etapow, dtlim);
						dtbuf[i] = ptcl[i].dt;
						// set_jp here, updated later when swap(i,j) is called
						gravity->set_jp(i, ptcl[i]);
					}
					prof.end_master(Profile::CORRECT);
#                 pragma omp master
					{
						prof.beg(Profile::SORT, true);
						// sort_ptcl_dtcache(nact, dtlim);
						sort_ptcl_dtcache(nact, dtlim, true);
						this->num_step += nact;
						this->num_bstep++;

						sh_tnext = dtbuf[0];
						sh_dtlim = calc_dtlim(sh_tnext);
						sh_nact  = count_nact(sh_tnext);
						prof.end(Profile::SORT);
					}
#if 0
#                 pragma omp barrier
					prof.beg_master(Profile::SET_JP, true);
#                 pragma omp for nowait
					for(int i=0; i<nact; i++){
						gravity->set_jp(i, ptcl[i]);
					}
					prof.end_master(Profile::SET_JP);
#endif
				}else{
#                 pragma omp master
					{
						prof.beg(Profile::CORRECT, true);
						for(int i=0; i<nact; i++){
							ptcl[i].correct(force[i], eta, etapow, dtlim);
							dtbuf[i] = ptcl[i].dt;
						}
						prof.end(Profile::CORRECT);
						prof.beg(Profile::SORT, true);
						sort_ptcl_dtcache(nact, dtlim);
						this->num_step += nact;
						this->num_bstep++;

						sh_tnext = dtbuf[0];
						sh_dtlim = calc_dtlim(sh_tnext);
						sh_nact  = count_nact(sh_tnext);
						prof.end(Profile::SORT);
						prof.beg(Profile::SET_JP, true);
						for(int i=0; i<nact; i++){
							gravity->set_jp(i, ptcl[i]);
						}
						prof.end(Profile::SET_JP);
					} // end master
				} // end if(nac > ...)
#             pragma omp barrier
#endif
				tsys_loc = tnext;
			} // while (tsys_loc < tt)
		} // end omp parallel
		tsys = tt;
	}
#endif

	void integrate_one_dtmax(){
		const double tt = tsys + dtmax;
#if !defined FAST_OMP_SYNC
		while(tsys < tt){
			integrate_one_block();
		}
#else
		integrate_one_dtmax_fast_omp();
#endif
		assert(tsys == tt);
	}

	void show_prof(FILE *fp=stderr){
		prof.show(nbody, num_step_tot, num_bstep_tot, fp);
	}

	void integrate(const double tcrit){
		prof.flush();
		prof.beg(Profile::TOTAL);
		while(tsys < tcrit){
			double t0 = Profile::wtime();
			integrate_one_dtmax();
			double t1 = Profile::wtime();
			print_stat(t1-t0);
		}
		prof.end(Profile::TOTAL);
		// prof.show(num_bstep);
		show_prof();
	}

	void print_stat(const double wtime, FILE *fp = stdout){
		const double energy = calc_energy(false);
		const double de_glo =((init_energy - energy) / init_energy);
		const double de_loc =((prev_energy - energy) / init_energy);
		num_step_tot  += num_step;
		num_bstep_tot += num_bstep;
		const double nact_avr = double(num_step) / double(num_bstep);
		const double Gflops = (((1.0e-9 * num_step) * nbody) * Particle::flops) / wtime;
		prof.beg(Profile::IO);
#if 0
		fprintf(fp, "t = %f\n", tsys);
		fprintf(fp, " steps: %ld %ld %ld %ld\n", num_bstep, num_step, num_bstep_tot, num_step_tot);
		fprintf(fp, " nact : %f\n", nact_avr);
		fprintf(fp, " de(local/global) : %+e %+e\n", de_loc, de_glo);
		fprintf(fp, " %f sec, %f Gflops\n", wtime, Gflops);
#else
		fprintf(stderr, "t = %f\n", tsys);
		fprintf(stderr, " steps: %ld %ld %ld %ld\n", num_bstep, num_step, num_bstep_tot, num_step_tot);
		fprintf(stderr, " nact : %f\n", nact_avr);
		fprintf(stderr, " de(local/global) : %+e %+e\n", de_loc, de_glo);
		fprintf(stderr, " %f sec, %f Gflops\n", wtime, Gflops);

        fprintf(fp, "%f %6ld %6ld %10ld %10ld  %8.2f  %+e  %+e  %f  %f \n", 
                tsys, 
                num_bstep, num_step, 
                num_bstep_tot, num_step_tot, 
                nact_avr, 
                de_loc, de_glo, 
                wtime, Gflops);
		fflush(fp);
		prof.end(Profile::IO);
#endif 
		prev_energy = energy;
		num_step  = 0;
		num_bstep = 0;
	}
};

