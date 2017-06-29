ifeq ($(OS),Windows_NT)
	CXX=g++
	CFLAGS=-Wall -O3 -std=c++11 -DNDEBUG -march=native
	ADI_FILES=ADI_serial.exe ADI_scalar.exe
	RW_FILES=RW_serial.exe RW_scalar.exe
else ifeq ($(USER),stud09)
	CXX=CC
	CFLAGS=-Wall -O3 -std=c++11 -DNDEBUG -march=core-avx2 -fno-alias -qopenmp
	ADI_FILES=ADI_serial.exe ADI_scalar.exe ADI_AVX.exe ADI_omp.exe
	RW_FILES=RW_serial.exe RW_scalar.exe RW_AVX.exe
else
	CXX=g++
	CFLAGS=-Wall -O3 -std=c++11 -DNDEBUG -march=native -fopenmp
	ADI_FILES=ADI_serial.exe ADI_scalar.exe ADI_AVX.exe ADI_omp.exe
	RW_FILES=RW_serial.exe RW_scalar.exe RW_AVX.exe
endif

HEADERS=timer.hpp prng_engine.hpp

all: ADI RW

adi: ADI

ADI: $(ADI_FILES)

rw: RW

RW: $(RW_FILES)

ADI_%.exe: ADI_%.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -o $@ $<

RW_%.exe: RW_%.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -o $@ $<

clean:
	rm -f *.exe
