#versao paralelizada
#Testing how the amount of processors changes the processing time
from multiprocessing import Process, Queue
import math
from random import uniform
from scipy import sqrt, pi
import time
import timeit
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def monteCarlo(npoints, nP):
	#number of tasks/works = nP

	resp = multiprocessing(f, npoints, nP)

	totalIn = resp

	nPI = 4.0 * totalIn / npoints

	#print ('Valor: ', nPI, '\nPrecisão: ', abs(nPI - pi))

def f(num):
	cont = 0
	
	for i in range(num):
		x = uniform(-1.0, 1.0)
		y = uniform(-1.0, 1.0)
		
		if (x*x + y*y) <= 1.0:
			cont += 1
	return cont
	
def multiprocessing(func, args, workers):
    begin_time = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        res = executor.submit(func, args)
    return (res.result())

def timeTestMonteCarlo():
	monteCarlo(npoints, nP[i])
	
######### main ##########
i = 0
npoints = 10000000
nP = [1, 2, 3, 4]
y = []

for i in range(len(nP)):
	t1 = timeit.Timer("timeTestMonteCarlo()", setup="from __main__ import timeTestMonteCarlo")
	times = t1.repeat(repeat=1, number=2)
	y.append(int((sum(times)/3.0)*1000)/1000.)

plt.plot(nP,y)
plt.show()
