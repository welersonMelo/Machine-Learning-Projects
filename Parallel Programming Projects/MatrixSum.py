#versao paralelizada
from multiprocessing import Process, Queue
import math
from scipy import sqrt, pi
from random import uniform
import time
import timeit
import matplotlib.pyplot as plt
import numpy as np

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def multiprocessing(workers):
    begin_time = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        res = executor.submit(np.add, a, b)
    return (res.result())

def sum2Matrix():
	multiprocessing(nP)
	
######### main ##########
#N x N matrix

nP = 5
print ('nP =',nP)
N = 1750
a = [[uniform(1,1000) for _ in range(N)] for _ in range(N)]
b = [[uniform(1,1000) for _ in range(N)] for _ in range(N)]

t1 = timeit.Timer("sum2Matrix()", setup="from __main__ import sum2Matrix")
times = t1.repeat(repeat=2, number=4)

print (sum(times)/6., 'segs')

####### Análise #########
''' Com as configurações acima:
	Para 1 processo  o tempo foi: 1.56 seg
	Para 2 processos o tempo foi: 1.59 seg
	Para 3 processos o tempo foi: 1.59 seg
	Para 4 processos o tempo foi: 1.60 seg
	Para 5 processos o tempo foi: 1.67 seg
	
	Usar multiprocesso nesse caso piorou o resultado encontrado com somente um processo.
	Ou seja, a função add já é paralelizada, por isso tentar paralelizar esse função só aumenta o tempo de processamento por overhead
'''
