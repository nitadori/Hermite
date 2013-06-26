#pragma once
#include <iosfwd>
#include <string>
#include <cassert>
#include <cmath>

struct Parameter{
	// typedef const char *string;
	typedef std::string string;

	string snapin;
	string snapout_base;
	string dumpin;
	string dumpout_base;

	int order;

	int snapid;
	int dumpid;

	double tend;
	double eta;
	double eta_s;
	double dtmax;
	double eps;
	double kt_for_eps;

	double log_interval;
	double snap_interval;
	double dump_interval;

	Parameter(){
		const double nan = +0.0/0.0;
		const double inf = 1.0/0.0;

		order = 0;

		snapid = 0;
		dumpid = 0;

		tend   = nan;
		eta    = nan;
		eta_s  = nan;
		dtmax  = nan;
		eps    = nan;
		kt_for_eps = nan;
		
		log_interval  = inf;
		snap_interval = inf;
		dump_interval = inf;
	}
	
	void read(std::istream &is);
	void print(std::ostream &os) const;

	void assert_check(const int porder) const {
		assert(snapin.length() >0 || dumpin.length() > 0);
		assert(order == porder);
		assert(tend  > 0);
		assert(eta   > 0);
		assert(eta_s > 0);
		assert(dtmax > 0);
		assert(log2(dtmax) == int(log2(dtmax)));
		assert(eps >= 0.0 || kt_for_eps >= 0.0);
	}
};
