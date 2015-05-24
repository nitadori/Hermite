VPATH = ../SRC
CXX = mpiFCCpx
CXXFLAGS = -Kfast,uxsimd,optmsg=2 -Xg -x32 -Nsrc
CXXFLAGS += -Kocl
CXXFLAGS += -Kopenmp
CXXFLAGS += -lfjcex

all: hpc-4th hpc-6th hpc-8th

cpu: cpu-4th cpu-6th cpu-8th

TOUCH: Makefile
	touch TOUCH

parameter.o: parameter.cpp parameter.h
	$(CXX) -O -Xg -c $< -o $@

cpu-4th: compile.cpp nbodysystem.h hermite4.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH $< -o $@

cpu-6th: compile.cpp nbodysystem.h hermite6.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH $< -o $@

cpu-8th: compile.cpp nbodysystem.h hermite8.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTH $< -o $@

cpu-8dd: compile.cpp nbodysystem.h hermite8dd.h TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTHDD $< -o $@


hpc-4th: hpc-4th.s hermite4-k.s parameter.o
	$(CXX) $(CXXFLAGS) $^ -o $@
#	cp $@ ../RUN

hpc-4th.s: compile.cpp nbodysystem.h hermite4.h hermite4-k.h TOUCH
	$(CXX) -S $(CXXFLAGS) -Knoswp,nounroll -DFOURTH -DHPC_ACE_GRAVITY -DFAST_OMP_SYNC $< -o $@

hermite4-k.s: hermite4-k.cpp hermite4-k.h hermite4.h TOUCH
	$(CXX) -S $(CXXFLAGS) -DFOURTH -DHPC_ACE_GRAVITY $< -o $@

hpc-6th: hpc-6th.s hermite6-k.s parameter.o
	$(CXX) $(CXXFLAGS)  $^ -o $@
#	cp $@ ../RUN

hpc-6th.s: compile.cpp nbodysystem.h hermite6.h hermite6-k.h TOUCH
	$(CXX) -S $(CXXFLAGS) -Knoswp,nounroll -DSIXTH -DHPC_ACE_GRAVITY -DFAST_OMP_SYNC $< -o $@

hermite6-k.s: hermite6-k.cpp hermite6-k.h hermite6.h TOUCH
	$(CXX) -S $(CXXFLAGS) -DSIXTH -DHPC_ACE_GRAVITY $< -o $@

hpc-8th: hpc-8th.s hermite8-k.s parameter.o
	$(CXX) $(CXXFLAGS)  $^ -o $@
#	cp $@ ../RUN

hpc-8th.s: compile.cpp nbodysystem.h hermite8.h hermite8-k.h TOUCH
	$(CXX) -S $(CXXFLAGS) -Knoswp,nounroll -DEIGHTH -DHPC_ACE_GRAVITY -DFAST_OMP_SYNC $< -o $@

hermite8-k.s: hermite8-k.cpp hermite8-k.h hermite8.h TOUCH
	$(CXX) -S $(CXXFLAGS) -DEIGHTH -DHPC_ACE_GRAVITY $< -o $@

hpc-8dd: hpc-8dd.s hermite8dd-k.s parameter.o
	$(CXX) $(CXXFLAGS)  $^ -o $@
#	cp $@ ../RUN

hpc-8dd.s: compile.cpp nbodysystem.h hermite8dd.h hermite8dd-k.h TOUCH
	$(CXX) -S $(CXXFLAGS) -DEIGHTHDD -DHPC_ACE_GRAVITY $< -o $@

hermite8dd-k.s: hermite8dd-k.cpp hermite8dd-k.h hermite8dd.h TOUCH
	$(CXX) -S $(CXXFLAGS) -DEIGHTHDD -DHPC_ACE_GRAVITY $< -o $@
