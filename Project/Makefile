CXX?=g++

CFLAGS=-Wall -O3 -std=c++11

all: diffusion2d_serial

diffusion2d_serial: diffusion2d_serial.cpp timer.hpp
	$(CXX) $(CFLAGS) -o main.exe $<

clean:
	rm -f main.exe
