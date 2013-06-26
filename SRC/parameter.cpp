#include "parameter.h"
#include <iostream>
#include <sstream>

void Parameter::read(std::istream &is){
	for(std::string line; std::getline(is, line); ){
		std::istringstream ss(line);
		std::string tag;
		ss >> tag;
		if(ss.fail()) continue;
		if(!isalpha(tag[0])) continue;;
#if 0
		static char rest[256];
		ss.read(rest, 255);
		std::cout << tag << " : " << rest << std::endl;
#endif

#define READ(name)                     \
		if(tag == std::string(#name)){ \
			ss >> name;                \
			continue;                  \
		}
#define READINV(name)                        \
		if(tag == std::string(#name "inv")){ \
			double temp;                     \
			ss >> temp;                      \
			name = 1.0 / temp;               \
			continue;                        \
		}

		READ(snapin);
		READ(snapout_base);
		READ(dumpin);
		READ(dumpout_base);

		READ(order);
		
		READ(snapid);
		READ(dumpid);

		READ(tend);
		READ(eta);
		READ(eta_s);
		READ(dtmax); READINV(dtmax);
		READ(eps);   READINV(eps);
		READ(kt_for_eps);

		READ(log_interval);
		READ(snap_interval);
		READ(dump_interval);
#undef READ
#undef READINV
		std::cerr << "unmatched tag : " << tag << std::endl;
	}
}

void Parameter::print(std::ostream &out) const {
#define PR(x) std::cout << #x << " : " << x << std::endl
	PR(snapin);
	PR(snapout_base);
	PR(dumpin);
	PR(dumpout_base);

	PR(order);

	PR(snapid);
	PR(dumpid);

	PR(tend);
	PR(eta);
	PR(eta_s);
	PR(dtmax);
	PR(eps);
	PR(kt_for_eps);

	PR(log_interval);
	PR(snap_interval);
	PR(dump_interval);
#undef PR
}

#ifdef TEST_MAIN
int main(){
	Parameter param;
	param.read(std::cin);
	std::cout << std::endl;
	param.print(std::cout);
	return 0;
}
#endif
