ifeq ($(OS),Windows_NT)
	CXX=g++
	CFLAGS=-Wall -O3 -std=c++11 -DNDEBUG -march=native
	ADI_FILES=ADI_serial.exe ADI_scalar.exe
	RW_FILES=RW_serial.exe RW_scalar.exe
else
	CXX=icpc
	CFLAGS=-Wall -O3 -std=c++11 -DNDEBUG -march=core-avx2
	ADI_FILES=ADI_serial.exe ADI_scalar.exe ADI_AVX.exe
	RW_FILES=RW_serial.exe RW_scalar.exe
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
