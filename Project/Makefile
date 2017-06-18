CXX?=g++

CFLAGS=-Wall -O3 -std=c++11 -DNDEBUG

HEADERS=timer.hpp prng_engine.hpp

all: ADI RW

adi: ADI

ADI: ADI_serial.exe

rw: RW

RW: RW_serial.exe RW_scalar.exe

ADI_%.exe: ADI_%.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -o $@ $<

RW_%.exe: RW_%.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -o $@ $<

clean:
	rm -f *.exe
