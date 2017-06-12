CXX?=g++

CFLAGS=-Wall -O3 -std=c++11

all: ADI_serial RW_serial

ADI_serial: ADI_serial.cpp timer.hpp
	$(CXX) $(CFLAGS) -o ADI.exe $<

RW_serial: RW_serial.cpp timer.hpp
	$(CXX) $(CFLAGS) -o RW.exe $<

clean:
	rm -f *.exe
