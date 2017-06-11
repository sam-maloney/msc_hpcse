from numpy import *
import matplotlib.pyplot as plt
import re

#files = ['bench_euler.128.stats.log', 'bench_euler.256.stats.log', 'bench_euler.1024.stats.log', 'bench_euler.2048.stats.log']
files = ['bench_euler.128.log', 'bench_euler.256.log', 'bench_euler.512.log' , 'bench_euler.1024.log','bench_euler.2048.log']

colors = {128 : 'b', 256 : 'r', 512 : 'y', 1024 : 'g', 2048 : 'm'}

serial = {}
threaded = {}
barrier  = {}
for fname in files:
    with open(fname) as f:
        myvar = None
        for line in f:
            if '== SERIAL ==' in line:
                myvar = serial
            if '== THREADED ==' in line:
                myvar = threaded
            if '== BARRIER ==' in line:
                myvar = barrier
            
            if myvar is not None:
                match = re.match('^Timing : (\d+) (\d+) (\d+.\d+)', line)
                if match:
                    ngrid    = int(match.group(1))
                    nthreads = int(match.group(2))
                    timing   = float(match.group(3))
                
                    if ngrid not in myvar:
                        myvar[ngrid] = {}
                    if not nthreads in myvar[ngrid]:
                        myvar[ngrid][nthreads] = []
                    myvar[ngrid][nthreads].append(timing)


plt.figure()

med1 = {}
mean1 = {}
for ngrid, d in sorted(serial.items()):
    if ngrid not in med1:
        med1[ngrid] = []
        mean1[ngrid] = []
    med1[ngrid].append( median(d[1]) )
    mean1[ngrid].append( mean(d[1]) )

for ngrid, d in sorted(threaded.items()):
    speedup = []
    threads = []
    for th, ti in sorted(d.items()):
        med = med1[ngrid] / median(ti)
        #q20 =  med - percentile(mean(d[1])/array(ti), 20)
        #q80 = -med + percentile(mean(d[1])/array(ti), 80)
        q20 =  med - percentile(mean1[ngrid]/array(ti), 20)
        q80 = -med + percentile(mean1[ngrid]/array(ti), 80)
        threads.append(th)
        speedup.append([ med, q20, q80 ])
        
    speedup = array(speedup)
    plt.errorbar(threads, speedup[:,0], yerr=(speedup[:,1],speedup[:,2]), fmt='--x'+colors[ngrid], label='Spawn+Join, N=%s' % ngrid)

for ngrid, d in sorted(barrier.items()):
    speedup = []
    threads = []
    for th, ti in sorted(d.items()):
        med = med1[ngrid] / median(ti)
        #q20 =  med - percentile(mean(d[1])/array(ti), 20)
        #q80 = -med + percentile(mean(d[1])/array(ti), 80)
        q20 =  med - percentile(mean1[ngrid]/array(ti), 20)
        q80 = -med + percentile(mean1[ngrid]/array(ti), 80)
        threads.append(th)
        speedup.append([ med, q20, q80 ])
    
    speedup = array(speedup)
    plt.errorbar(threads, speedup[:,0], yerr=(speedup[:,1],speedup[:,2]), fmt='-x'+colors[ngrid], label='Barrier, N=%s' % ngrid)

plt.xlim(xmin=0)

xmin, xmax = plt.xlim()
perfect = linspace(xmin, xmax)
plt.plot(perfect, perfect, '-k')

plt.ylabel('Speedup')
plt.xlabel('# threads')
plt.legend(loc='best')

plt.savefig('strong_scaling.pdf')
plt.show()
