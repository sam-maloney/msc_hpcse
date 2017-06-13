CXX?=g++

CFLAGS=-Wall -O3 -std=c++11 -DNDEBUG

all: ADI_serial RW_serial

adi: ADI

ADI: ADI_serial

rw: RW

RW: RW_serial

ADI_serial: ADI_serial.cpp timer.hpp
	$(CXX) $(CFLAGS) -o ADI.exe $<

RW_serial: RW_serial.cpp timer.hpp
	$(CXX) $(CFLAGS) -o RW.exe $<

clean:
	rm -f *.exe
