ifeq ($(OS),Windows_NT)
	CXX=g++
	CFLAGS=-Wall -O3 -std=c++11 -DNDEBUG -march=native
	ADI_FILES=ADI_serial.exe ADI_scalar.exe
	RW_FILES=RW_serial.exe RW_scalar.exe
else ifeq ($(USER),stud09)
	CXX=CC
	CFLAGS=-Wall -O3 -std=c++11 -DNDEBUG -march=core-avx2 -fno-alias -qopenmp -mkl
	ADI_FILES=ADI_serial.exe ADI_scalar.exe ADI_AVX.exe ADI_omp.exe
	RW_FILES=RW_serial.exe RW_scalar.exe RW_AVX.exe RW_omp.exe
	PAT_FILES=ADI_pat_serial.exe ADI_pat_scalar.exe ADI_pat_AVX.exe ADI_pat_omp.exe
else
	CXX=g++
	CFLAGS=-Wall -O3 -std=c++11 -DNDEBUG -march=native -fopenmp
	ADI_FILES=ADI_serial.exe ADI_scalar.exe ADI_AVX.exe ADI_omp.exe
	RW_FILES=RW_serial.exe RW_scalar.exe RW_AVX.ex RW_omp.exe
endif

HEADERS=timer.hpp

all: ADI RW

adi: ADI

ADI: $(ADI_FILES)

rw: RW

RW: $(RW_FILES)

ADI_pat_%.exe: ADI_%.exe ADI_%.o
	pat_build -w -T _ZN11Diffusion2D14run_simulationEd $< $@

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CFLAGS)

ADI_%.exe: ADI_%.cpp $(HEADERS)
	$(CXX) -o $@ $< $(CFLAGS)

RW_%.exe: RW_%.cpp $(HEADERS)
	$(CXX) -o $@ $< $(CFLAGS)

craypat: $(PAT_FILES)
	export PAT_RT_PERFCTR="PAPI_TOT_CYC,PAPI_L3_TCM,PAPI_L2_TCM,PAPI_L1_TCM"

clean:
	rm -f *.exe
