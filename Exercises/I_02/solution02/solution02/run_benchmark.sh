#!/bin/sh

#CORES="1 2 4 6 8 10 12 16 18 20 22 24"
CORES="1 2 4 6 8 10 12 24"

#N=2048
#DT=0.00000001

#N=1024
#DT=0.00000001

N=256
DT=0.000001

#N=128
#DT=0.00001

REPETITIONS=10

numactl -show

for i in `seq $REPETITIONS`
do
    echo "== THREADED =="
    for n in $CORES
    do
        ./diffusion2d_threaded 1 2 $N $DT $n
    done
    echo "== BARRIER =="
    for n in $CORES
    do
        ./diffusion2d_barrier 1 2 $N $DT $n
    done
done
