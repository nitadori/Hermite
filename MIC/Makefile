VPATH = ../SRC
CXX = icpc
CXXFLAGS = -Wall -O -mmic -openmp

OBJS = parameter.o

all: cpu-4th cpu-6th cpu-8th

TOUCH: Makefile
	touch TOUCH

parameter.o: parameter.cpp parameter.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

cpu-4th: compile.cpp nbodysystem.h hermite4.h $(OBJS) TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH $< $(OBJS) -o $@

cpu-6th: compile.cpp nbodysystem.h hermite6.h $(OBJS) TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH $< $(OBJS) -o $@

cpu-8th: compile.cpp nbodysystem.h hermite8.h $(OBJS) TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTH $< $(OBJS) -o $@

mic-4th: compile.cpp nbodysystem.h mic-hard.h hermite4.h hermite4-mic.h $(OBJS) TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH -DMIC_GRAVITY $< $(OBJS) -o $@

mic-4th.s: compile.cpp nbodysystem.h mic-hard.h hermite4.h hermite4-mic.h TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH -DMIC_GRAVITY -S $<  -o $@

mic-6th: compile.cpp nbodysystem.h mic-hard.h hermite6.h hermite6-mic.h $(OBJS) TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH -DMIC_GRAVITY $< $(OBJS) -o $@

mic-6th.s: compile.cpp nbodysystem.h mic-hard.h hermite6.h hermite6-mic.h TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH -DMIC_GRAVITY -S $<  -o $@

mic-8th: compile.cpp nbodysystem.h mic-hard.h hermite8.h hermite8-mic.h $(OBJS) TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTH -DMIC_GRAVITY $< $(OBJS) -o $@

mic-8th.s: compile.cpp nbodysystem.h mic-hard.h hermite8.h hermite8-mic.h TOUCH
	$(CXX) $(CXXFLAGS) -DEIGHTH -DMIC_GRAVITY -S $<  -o $@

