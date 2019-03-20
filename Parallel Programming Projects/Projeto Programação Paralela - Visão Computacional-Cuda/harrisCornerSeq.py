# para rodar faça: python3 harrisCornerSeq.py imagemName.extensao
import numpy as np
import math
import imgLib 
import sys
import baselib
import matplotlib

#################### MAIN #################################
fileName = sys.argv[1]
#img = cv.imread(fileName, cv.IMREAD_UNCHANGED)
img = baselib.imread(fileName)
#imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgG = baselib.rgb2gray(img)

h, w = baselib.size(imgG)

#Gerando Gaussian mask 3X3
g3Mask = imgLib.gaussianMaskGenerator()

###GaussianMask###
imgG = imgLib.convolve(imgG, g3Mask)

###SobelMask 5X5 ###
s5MaskX = imgLib.sobelMaskGenerator()

#### Aplicando Sobel Operator a partir da convolução ####
Ix = imgLib.convolve(imgG, s5MaskX)
Iy = imgLib.convolve(imgG, np.transpose(s5MaskX))

#cv.imwrite('response Iy.png', Iy)
#cv.imwrite('response Ix.png', Ix)

###Produto element wise de matriz: ou seja C[i][j] = A[i][j]*B[i][j]###
Ix2 = np.multiply(Ix,Ix)
Iy2 = np.multiply(Iy,Iy)
Ixy = np.multiply(Ix,Iy)

Ix2 = imgLib.convolve(Ix2, g3Mask)
Iy2 = imgLib.convolve(Iy2, g3Mask)
Ixy = imgLib.convolve(Ixy, g3Mask)

### response calc ###
response = imgLib.calcResponse(Ix2, Iy2, Ixy, h, w)

###normalizing###
#response = cv.normalize(response, response, 0.0, 1000.0, cv.NORM_MINMAX, cv.CV_32FC1)
rmax, rmin = response.max(), response.min()
response = ((response - rmin)/(rmax-rmin)) * 1000

###nonMaxima###
# Lista de keypoint sendo a uma tupla da posicao na imagem (i, j) onde i é a linha e j a coluna
keyPointsList = imgLib.nonMaxima(response)

###put KeyPoints in the image###
r = int(max(h, w)*0.015)

for point in keyPointsList:
	#cv.circle(img, (point[1],point[0]) , r, (0,0,255), -1)
	i, j = point
	
	for u in range(-1,2):
		for v in range(-1,2):
			x = min(max(j + v, 0), w-1)
			y = min(max(i + u, 0), h-1)
			img[y][x] = (255,0,0)

#baselib.imshow(img)
matplotlib.image.imsave(fileName+'Keypoints.png', img)
#cv.imwrite('KeyPoints'+fileName, img)

#Jogando coordenadas dos keypoins num txt
with open('keyPoints'+fileName+'.txt', 'w') as f:
    for item in keyPointsList:
        f.write('('+str(item[0]) +', '+ str(item[1])+')'+ "\n")
