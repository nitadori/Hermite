VPATH = ../SRC
#CXX = /usr/local/bin/g++
CXX = g++
CXXFLAGS = -Wall -O2
CXXFLAGS += -march=core-avx2
#CXXFLAGS += -march=bdver2
CXXFLAGS += -fopenmp
CXXFLAGS += -DFAST_OMP_SYNC
#CXXFLAGS += -ffast-math

all: cpu-4th cpu-6th cpu-8th avx2-4th avx2-6th avx2-8th
avx2: avx2-4th avx2-6th avx2-8th

TOUCH: Makefile
	touch TOUCH

cpu-4th: compile.cpp nbodysystem.h hermite4.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH $< parameter.o -o $@

cpu-6th: compile.cpp nbodysystem.h hermite6.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH $< parameter.o -o $@

cpu-6th.s: compile.cpp nbodysystem.h hermite6.h TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH -S $< -o $@

cpu-8th: compile.cpp nbodysystem.h hermite8.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTH $< parameter.o -o $@

avx2-4th: compile.cpp nbodysystem.h hermite4.h hermite4-avx.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH -DAVX_GRAVITY  $< parameter.o -o $@

avx2-4th.s: compile.cpp nbodysystem.h hermite4.h hermite4-avx.h TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH -DAVX_GRAVITY  -S $< -o $@

avx2-6th: compile.cpp nbodysystem.h hermite6.h hermite6-avx.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH -DAVX_GRAVITY  $< parameter.o -o $@

avx2-6th.s: compile.cpp nbodysystem.h hermite6.h hermite6-avx.h TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH -DAVX_GRAVITY  -S $< -o $@

avx2-8th: compile.cpp nbodysystem.h hermite8.h hermite8-avx.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTH -DAVX_GRAVITY  $< parameter.o -o $@

avx2-8th.s: compile.cpp nbodysystem.h hermite8.h hermite8-avx.h TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTH -DAVX_GRAVITY  -S $< -o $@

parameter.o: parameter.cpp parameter.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
