import numpy as np
import math
import imgLib 
import sys
import baselib
import matplotlib
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor
import time 

start = time.time()

#################### MAIN #################################
fileName = sys.argv[1]
#img = cv.imread(fileName, cv.IMREAD_UNCHANGED)
img = baselib.imread(fileName)

imgG = baselib.rgb2gray(img)
h, w = baselib.size(imgG)

#Gerando Gaussian mask 3X3
g3Mask = imgLib.gaussianMaskGenerator()

###GaussianMask###
imgG = imgLib.convolveGPU(imgG, g3Mask)

###SobelMask 5X5 ###
s5MaskX = imgLib.sobelMaskGenerator()

#### Aplicando Sobel Operator a partir da convolução ####
Ix = imgLib.convolveGPU(imgG, s5MaskX)
Iy = imgLib.convolveGPU(imgG, np.transpose(s5MaskX))

#cv.imwrite('response Iy.png', Iy)
#cv.imwrite('response Ix.png', Ix)

###Produto element wise de matriz: ou seja C[i][j] = A[i][j]*B[i][j]###
Ix2 = np.multiply(Ix,Ix)
Iy2 = np.multiply(Iy,Iy)
Ixy = np.multiply(Ix,Iy)

Ix2 = imgLib.convolveGPU(Ix2, g3Mask)
Iy2 = imgLib.convolveGPU(Iy2, g3Mask)
Ixy = imgLib.convolveGPU(Ixy, g3Mask)

### response calc###
response = imgLib.calcResponse(Ix2, Iy2, Ixy, h, w)

###normalizing###
rmax, rmin = response.max(), response.min()
response = ((response - rmin)/(rmax-rmin)) * 1000

###nonMaxima Paralelizado com processos CPU ###
# Lista de keypoint sendo a uma tupla da posicao na imagem (i, j) onde i é a linha e j a coluna 
nP = 6 #numero de processos
workers = nP
func = imgLib.nonMaxima
args = response 

with ProcessPoolExecutor(max_workers=workers) as executor:
	res = executor.submit(func, args)

keyPointsList = res.result()

###put KeyPoints in the image###

for point in keyPointsList:
	#cv.circle(img, (point[1],point[0]) , r, (0,0,255), -1)
	i, j = point
	
	for u in range(-1,2):
		for v in range(-1,2):
			x = min(max(j + v, 0), w-1)
			y = min(max(i + u, 0), h-1)
			img[y][x] = (255,0,0)


matplotlib.image.imsave(fileName+'Keypoints.png', img)

#Jogando coordenadas dos keypoins num txt
with open('keyPoints'+fileName+'.txt', 'w') as f:
    for item in keyPointsList:
        f.write('('+str(item[0]) +', '+ str(item[1])+')'+ "\n")

print('Tempo Parte final: ',time.time() - start)
