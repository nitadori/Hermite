VPATH = ../SRC
CXX = /usr/local/bin/g++
#CXX = g++
CXXFLAGS = -Wall -O2
#CXXFLAGS += -march=core2 -mavx
CXXFLAGS += -march=bdver2
CXXFLAGS += -fopenmp
CXXFLAGS += -DFAST_OMP_SYNC
#CXXFLAGS += -ffast-math

all: cpu-4th cpu-6th cpu-8th avx-4th avx-6th avx-8th
avx: avx-4th avx-6th avx-8th

TOUCH: Makefile
	touch TOUCH

cpu-4th: compile.cpp nbodysystem.h hermite4.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH $< parameter.o -o $@

cpu-6th: compile.cpp nbodysystem.h hermite6.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH $< parameter.o -o $@

cpu-6th.s: compile.cpp nbodysystem.h hermite6.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH -S $< parameter.o -o $@

cpu-8th: compile.cpp nbodysystem.h hermite8.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTH $< parameter.o -o $@

cpu-8dd: compile.cpp nbodysystem.h hermite8dd.h TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTHDD $< -o $@

avx-4th: compile.cpp nbodysystem.h hermite4.h hermite4-avx.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH -DAVX_GRAVITY $< parameter.o -o $@

avx-4th.s: compile.cpp nbodysystem.h hermite4.h hermite4-avx.h TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH -DAVX_GRAVITY -S $< -o $@

avx-6th: compile.cpp nbodysystem.h hermite6.h hermite6-avx.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH -DAVX_GRAVITY $< parameter.o -o $@

avx-6th.s: compile.cpp nbodysystem.h hermite6.h hermite6-avx.h TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH -DAVX_GRAVITY -S $< -o $@

avx-8th: compile.cpp nbodysystem.h hermite8.h hermite8-avx.h parameter.o TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTH -DAVX_GRAVITY $< parameter.o -o $@

avx-8th.s: compile.cpp nbodysystem.h hermite8.h hermite8-avx.h TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTH -DAVX_GRAVITY -S $< -o $@

avx-8dd: compile.cpp nbodysystem.h hermite8dd.h hermite8dd-avx.h TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTHDD -DAVX_GRAVITY $< -o $@

avx-8dd.s: compile.cpp nbodysystem.h hermite8dd.h hermite8dd-avx.h TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTHDD -DAVX_GRAVITY -S -mfma4 $< -o $@

parameter.o: parameter.cpp parameter.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
