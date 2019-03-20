import numpy as np
import math
import baselib
################################################################
def calcResponse(Ix2, Iy2, Ixy, h, w):
	response = []
	for i in range(h):
		aux = []
		for j in range(w):
			det = (Ix2[i][j]*Iy2[i][j]) - (Ixy[i][j]*Ixy[i][j])
			trace = Ix2[i][j] + Iy2[i][j]
			val = det - (0.04 * (trace*trace))
			aux.append(0 if val <= 1 else val)
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
    
    return [a1, a2, a3, -a2, -a1] 

################################################################
def gaussianMaskGenerator():
	return [[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]

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
