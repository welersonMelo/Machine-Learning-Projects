import numpy as np
import math
import baselib

from pycuda import driver, compiler, tools

import pycuda.autoinit

import pycuda.gpuarray as gpuarray

from pycuda.reduction import ReductionKernel

################################################################
def calcResponse(Ix2, Iy2, Ixy, h, w):
	response = []
	for i in range(h):
		aux = []
		for j in range(w):
			det = (Ix2[i][j]*Iy2[i][j]) - (Ixy[i][j]*Ixy[i][j])
			trace = Ix2[i][j] + Iy2[i][j]
			val = det - (0.04 * (trace*trace))
			aux.append(0 if val <= 0 else val)
		response.append(aux)
	
	return np.asarray(response, np.float32)

################################################################
def convolve(img, kernel):
	Sy, Sx = baselib.size(img)
	
	a = len(kernel)
	b = len(kernel[0])
	a2 = int(a/2)
	b2 = int(b/2)
	
	outAux = []
	
	for i in range(Sy):
		aux = []
		for j in range(Sx):
			g = 0.0
			for s in range(a):
				for t in range(b):
					x = j+t-a2
					y = i+s-b2
					x = min(max(x, 0), Sx-1)
					y = min(max(y, 0), Sy-1)
					g += (kernel[s][t] * img[y][x])
			aux.append(g)
		outAux.append(aux)
	
	return np.asarray(outAux, np.float32)

################## generating Sobel Mask #######################
def sobelMaskGenerator():
    a1 = np.asarray([2,2,4,2,2])
    a2 = np.asarray([1,1,2,1,1])
    a3 = np.asarray([0,0,0,0,0])
    
    return np.asarray([a1, a2, a3, -a2, -a1])

################################################################
def gaussianMaskGenerator():
	d = 159;
	return np.asarray([[2/d,4/d,5/d,4/d,2/d], [4/d,9/d,12/d,9/d,4/d], [5/d,12/d,15/d,12/d,5/d] ,[4/d,9/d,12/d,9/d,4/d] ,[2/d,4/d,5/d,4/d,2/d]])	

################################################################
def nonMaxima(img):
	Sy, Sx = baselib.size(img)
	
	maskSize = 21
	
	a = maskSize
	b = maskSize
	a2 = int(a/2)
	b2 = int(b/2)
	
	keyPoints = [(0,0)]
	
	for i in range(Sy):
		for j in range(Sx):
			
			p = img[i][j]
			if p < 100:
				continue
			
			isMax = True
			
			for s in range(a):
				for t in range(b):
					x = j+t-a2
					y = i+s-b2
					x = min(max(x, 0), Sx-1)
					y = min(max(y, 0), Sy-1)
					if p < img[y][x]:
						isMax = False
						break
			if isMax:
				keyPoints.append((i, j))
		
	return keyPoints


def convolveGPU(img, kernel):
	
	kernel_code_template = """
	__global__ void matrixMult(float *img, float *B, float *C)
	{
		int a = 5, b = 5;
		int a2 = a/2, b2 = b/2;
		float g = 0;
		int j = threadIdx.x;
		int i = threadIdx.y;
		int n = %(N)s, m =  %(M)s;
		
		for (int s = 0; s < a; s+=1){
			for (int t = 0; t < b; t+=1) {
				int x = j+t-a2;
				int y = i+s-b2;
				x = min(max(x,0), n-1);
				y = min(max(y,0), m-1);
				g += B[s*a + t] * img[y*n + x];
			}
		}
		C[i*n + j] = g;
	}
	"""
	
	m, n = baselib.size(img)
	
	emp = [[0.0 for _ in range(n)] for _ in range(m)]
	# Transferindo os arrays para o device.
	img_gpu = gpuarray.to_gpu(np.asarray(img, np.float32)) 
	b_gpu = gpuarray.to_gpu(np.asarray(kernel, np.float32))
	c_gpu = gpuarray.empty(np.asarray(emp, np.float32))

	kernel_code = kernel_code_template % {
		'N': n,
		'M': m
		}
	mod = compiler.SourceModule(kernel_code)
	convolution = mod.get_function("convolution")
	
	matrixMult(img_gpu, b_gpu, c_gpu, block = (m, n, 1))
	
	return c_gpu.get()
	
	

