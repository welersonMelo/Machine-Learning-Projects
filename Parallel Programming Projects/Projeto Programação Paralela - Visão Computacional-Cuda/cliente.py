import rpyc
import sys
import numpy as np
import math
import imgLib 
import sys
import baselib
import matplotlib

c = rpyc.connect("localhost", 18861, config = {'allow_all_attrs' : True})

fileName = sys.argv[1]
#img = cv.imread(fileName, cv.IMREAD_UNCHANGED)
img = baselib.imread(fileName)
img2 = img.copy()
#imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgG = baselib.rgb2gray(img2)

#servidor
h, w = c.root.get_size(imgG) 

#Gerando Gaussian mask 3X3
g3Mask = imgLib.gaussianMaskGenerator()

###GaussianMask###
imgG = imgLib.convolve(imgG, g3Mask) #calculei aqui, pois no servidor da timeout

###SobelMask 5X5 ###
s5MaskX = imgLib.gaussianMaskGenerator() #ate aqui, okay, e se calcular isso no servirdor ele da um erro de pickling disabled

Ix = imgLib.convolve(imgG, s5MaskX)
Ixcopy = Ix
Iy = imgLib.convolve(imgG, np.transpose(s5MaskX))

Ix2 = np.multiply(Ix,Ix) #se passar para o servirdor da um erro de recursão gigante
Iy2 = np.multiply(Iy,Iy)
Ixy = np.multiply(Ix,Iy)

Ix2 = imgLib.convolve(Ix2, g3Mask)
Iy2 = imgLib.convolve(Iy2, g3Mask)
Ixy = imgLib.convolve(Ixy, g3Mask)

response = imgLib.calcResponse(Ix2, Iy2, Ixy, h, w) # aqui tbm da timeout quando passa para o servidor

rmax, rmin = response.max(), response.min()

#servidor
response = c.root.get_Response(response, rmax, rmin) # aqui ele esta dando algum warning q ñ sei o motivo

keyPointsList = imgLib.nonMaxima(response)

r = int(max(h, w)*0.015)

#img2 = c.root.get_keyPoint(keyPointsList, w, h, img2)

#Coloquei essa função no servidor, mas tbm da timeout
for point in keyPointsList:     
	#cv.circle(img, (point[1],point[0]) , r, (0,0,255), -1)
	i, j = point
	
	for u in range(-1,2):
		for v in range(-1,2):
			x = min(max(j + v, 0), w-1)
			y = min(max(i + u, 0), h-1)
			img2[y][x] = (255,0,0)


#baselib.imshow(img)
matplotlib.image.imsave(fileName+'Keypoints.png', img2)
#cv.imwrite('KeyPoints'+fileName, img2)

#Jogando coordenadas dos keypoins num txt
with open('keyPoints'+fileName+'.txt', 'w') as f:
    for item in keyPointsList:
        f.write('('+str(item[0]) +', '+ str(item[1])+')'+ "\n")


#print(c.root.get_year())
print(c.root.last_year)
#print(c.root.get_question())
