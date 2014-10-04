VPATH = ../SRC
CXX = g++
CXXFLAGS = -Wall -O2
CXXFLAGS += -march=core2 -mavx
CXXFLAGS += -fopenmp
CXXFLAGS += -I /usr/local/cuda/include -I/usr/local/cuda/samples/common/inc
NVCC = /usr/local/cuda/bin/nvcc
NVFLAGS = -O -arch=sm_35 -Xptxas -v
LIBCUDA = -L /usr/local/cuda/lib64 -lcudart

all : gpu-4th gpu-6th

ptx : hermite4-titan.ptx hermite6-titan.ptx

cubin : hermite4-titan.cubin hermite6-titan.cubin

sass : hermite4-titan.sass hermite6-titan.sass

parameter.o: parameter.cpp parameter.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

gpu-4th: compile.cpp nbodysystem.h hermite4.h hermite4-titan.h parameter.o hermite4-titan.o TOUCH
	$(CXX) $(CXXFLAGS) -DFOURTH -DCUDA_TITAN $< parameter.o hermite4-titan.o $(LIBCUDA) -o $@
	cp $@ ../RUN

gpu-6th: compile.cpp nbodysystem.h hermite6.h hermite6-titan.h parameter.o hermite6-titan.o TOUCH
	$(CXX) $(CXXFLAGS) -DSIXTH -DCUDA_TITAN $< parameter.o hermite6-titan.o $(LIBCUDA) -o $@
	cp $@ ../RUN

hermite4-titan.o: hermite4-titan.cu hermite4-titan.h hermite4.h
	$(NVCC) $(NVFLAGS) -c $<

hermite6-titan.o: hermite6-titan.cu hermite6-titan.h hermite6.h
	$(NVCC) $(NVFLAGS) -c $<

hermite4-titan.ptx: hermite4-titan.cu hermite4-titan.h hermite4.h
	$(NVCC) $(NVFLAGS) -ptx $<

hermite6-titan.ptx: hermite6-titan.cu hermite6-titan.h hermite6.h
	$(NVCC) $(NVFLAGS) -ptx $<

hermite4-titan.cubin: hermite4-titan.cu hermite4-titan.h hermite4.h
	$(NVCC) $(NVFLAGS) -cubin $< |& c++filt

hermite6-titan.cubin: hermite6-titan.cu hermite6-titan.h hermite6.h
	$(NVCC) $(NVFLAGS) -cubin $< |& c++filt

hermite4-titan.sass: hermite4-titan.cubin
	/usr/local/cuda/bin/cuobjdump -sass $< | c++filt > $@

hermite6-titan.sass: hermite6-titan.cubin
	/usr/local/cuda/bin/cuobjdump -sass $< | c++filt > $@


