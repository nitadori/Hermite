VPATH = ../SRC
CXX = FCCpx
CXXFLAGS = -Kfast -Xg -x64
#CXXFLAGS += Koptmsg=2 -Nsrc
#CXXFLAGS += -Knoswp,nounroll
CXXFLAGS += -Kopenmp
CXXFLAGS += -Kocl
CXXFLAGS += -lfjcex

all: mx-4th mx-6th mx-8th

cpu: cpu-4th cpu-6th cpu-8th

TOUCH: Makefile
	touch TOUCH

parameter.o: parameter.cpp parameter.h
	$(CXX) -O -Xg -c $< -o $@

cpu-4th: compile.cpp nbodysystem.h hermite4.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH parameter.o $< -o $@

cpu-6th: compile.cpp nbodysystem.h hermite6.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH parameter.o $< -o $@

cpu-8th: compile.cpp nbodysystem.h hermite8.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTH parameter.o $< -o $@

mx-4th: mx-4th.s hermite4-mx.s parameter.o
	$(CXX) $(CXXFLAGS)  $^ -o $@
	cp $@ ../RUN

mx-4th.s: compile.cpp nbodysystem.h hermite4.h hermite4-mx.h TOUCH
	$(CXX) -S $(CXXFLAGS) -DFOURTH -DMX_GRAVITY -DFAST_OMP_SYNC $< -o $@

hermite4-mx.s: hermite4-mx.cpp hermite4-mx.h hermite4.h TOUCH
	$(CXX) -S $(CXXFLAGS) -DFOURTH -DMX_GRAVITY $< -o $@

mx-6th: mx-6th.s hermite6-mx.s parameter.o
	$(CXX) $(CXXFLAGS)  $^ -o $@
	cp $@ ../RUN

mx-6th.s: compile.cpp nbodysystem.h hermite6.h hermite6-mx.h TOUCH
	$(CXX) -S $(CXXFLAGS) -DSIXTH -DMX_GRAVITY -DFAST_OMP_SYNC $< -o $@

hermite6-mx.s: hermite6-mx.cpp hermite6-mx.h hermite6.h TOUCH
	$(CXX) -S $(CXXFLAGS) -DSIXTH -DMX_GRAVITY $< -o $@

mx-8th: mx-8th.s hermite8-mx.s parameter.o
	$(CXX) $(CXXFLAGS)  $^ -o $@
	cp $@ ../RUN

mx-8th.s: compile.cpp nbodysystem.h hermite8.h hermite8-mx.h TOUCH
	$(CXX) -S $(CXXFLAGS) -DEIGHTH -DMX_GRAVITY -DFAST_OMP_SYNC $< -o $@

hermite8-mx.s: hermite8-mx.cpp hermite8-mx.h hermite8.h TOUCH
	$(CXX) -S $(CXXFLAGS) -Krestp=all -DEIGHTH -DMX_GRAVITY $< -o $@

