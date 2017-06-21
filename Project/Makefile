CXX=icpc

CFLAGS=-Wall -O3 -std=c++11 -DNDEBUG -march=core-avx2

HEADERS=timer.hpp prng_engine.hpp

all: ADI RW

adi: ADI

ADI: ADI_serial.exe ADI_scalar.exe ADI_AVX.exe

rw: RW

RW: RW_serial.exe RW_scalar.exe

ADI_%.exe: ADI_%.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -o $@ $<

RW_%.exe: RW_%.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -o $@ $<

clean:
	rm -f *.exe
