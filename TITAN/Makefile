VPATH = ../SRC
CXX = g++
CXXFLAGS = -Wall -O2
CXXFLAGS += -march=core2 -mavx
CXXFLAGS += -fopenmp
CXXFLAGS += -I /usr/local/cuda/include -I/usr/local/cuda/samples/common/inc
NVCC = /usr/local/cuda/bin/nvcc
NVFLAGS = -O -arch=sm_35
LIBCUDA = -L /usr/local/cuda/lib64 -lcudart

all : gpu-4th

parameter.o: parameter.cpp parameter.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

gpu-4th: compile.cpp nbodysystem.h hermite4.h hermite4-titan.h parameter.o hermite4-titan.o TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH -DCUDA_TITAN $< parameter.o hermite4-titan.o $(LIBCUDA) -o $@
	cp $@ ../RUN

hermite4-titan.o: hermite4-titan.cu hermite4-titan.h hermite4.h
	$(NVCC) $(NVFLAGS) -c $<


