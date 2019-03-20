########################################
#
# Nome: WELERSON AUGUSTO LINO DE JESUS MELO
# Matricula: 201600017230
# E-mail: welerson.a.melo@gmail.com
#
# Nome: THIAGO JOSE SANDES MELO
# Matricula: 201600092557	
# E-mail: thiago_sandes@outlook.com
#
########################################

import matplotlib 
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Q2
def imread(file):
    img = mpimg.imread(file)
    return np.asarray(img, np.uint8)

#Q3
def nchannels(img):
	if len(img.shape) == 2:
		return 1
	return len(img.shape)

#Q4
def size(img):
		x, y = img.shape[0], img.shape[1]
		return [y, x]

#Q5
def rgb2gray(img):
	x, y = size(img)
	ans = np.zeros((y, x), np.uint8)
	i = -1
	for col in img:
		i+=1
		for j in range(len(col)):
			p = col[j]
			ans[i][j] = (p[0]*0.299 + p[1]*0.587 + p[2]*0.114)
			
	return ans

#Q6
def imreadgray(file):
	img = imread(file)
	if nchannels(img) == 1:
		return img
	return rgb2gray(img)

#Q7
def imshow(img):
	if nchannels(img) == 1:
		plt.imshow(img, cmap='gray', vmin = 0, vmax = 255)
	else:
		plt.imshow(img, vmin = 0, vmax = 255)
	plt.show()

#Q8
def thresh(img, limiar):
	imgOut = []
	if nchannels(img) == 1:
		for col in img:
			imgOut.append([255 if p >= limiar else 0 for p in col])
	else:
		for col in img:
			line = []
			for p in col:
				if p[0] >= limiar[0]:
					a0 = 255
				else:
					a0 = 0
				if p[1] >= limiar[1]:
					a1 = 255
				else:
					a1 = 0
				if p[2] >= limiar[2]:
					a2 = 255
				else:
					a2 = 0
				line.append([a0, a1, a2,])
			imgOut.append(line)
	return np.asarray(imgOut, np.uint8)

#Q9
def negative(img):
	return 255 - img;

#Q10
def contrast(img, r, m):
	out = r * (img - m) + m
	out = [[255 if x > 255 else x for x in arr] for arr in out]
	out = [[0 if x < 0 else x for x in arr] for arr in out]
	
	return np.asarray(out, np.uint8)

#Q11
def hist(img):
	if nchannels(img) == 1:
		h = [0 for _ in range(256)]
		for col in img:
			for p in col:
				h[p]+=1
		return h
	else:
		h = [[0 for _ in range(256)] for _ in range(3)]
		for col in img:
			for p in col:
				h[0][p[0]]+=1
				h[1][p[1]]+=1
				h[2][p[2]]+=1
		return h

#Q12 e Q13
def showhist(h, Bin = 1):
	if len(h) != 3:
		x = [sum(h[i : i+Bin])/Bin*1. for i in range(0, 256, Bin)]
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_xlim(0, len(x))	
		ax.stem([i for i in range(len(x))], x)
		plt.show()
	else:
		x = []
		x.append([sum(h[0][i : i+Bin])/Bin*1. for i in range(0, 256, Bin)])
		x.append([sum(h[1][i : i+Bin])/Bin*1. for i in range(0, 256, Bin)])
		x.append([sum(h[2][i : i+Bin])/Bin*1. for i in range(0, 256, Bin)])
		
		fig = plt.figure()
		ax1 = fig.add_subplot(111, projection = '3d')

		xpos = [i for i in range(len(x[0]))] + [i for i in range(len(x[0]))] + [i for i in range(len(x[0]))]
		ypos = [1 for _ in range(len(x[0]))] + [2 for _ in range(len(x[0]))] + [3 for _ in range(len(x[0]))]
		zpos = [0 for i in range(len(x[0])*3)]

		dx = np.ones(len(x[0])*3)
		dy = np.ones(len(x[0])*3)
		dz = x[0]+x[1]+x[2]

		ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = 'red')
		plt.show()

#Q14
def histeq(img):
	h = hist(img)
	Sx, Sy = size(img)
	n = Sx * Sy * 1.0
	p = [h[i]/n for i in range(256)]
	cdf = [p[0]]
	for i in range(1, 256):
		cdf.append((cdf[i-1] + p[i]))
	
	cdf = [x*255 for x in cdf]
	
	cdf = [255 if x > 255 else int(x) for x in cdf]
	cdf = [0 if x < 0 else int(x) for x in cdf]
	
	out = np.zeros((Sy, Sx), np.uint8)
	
	for i in range(Sy):
		for j in range(Sx):
			out[i][j] = cdf[img[i][j]]
	return out

#Q15
def convolve(img, kernel):
	Sx, Sy = size(img)
	
	a = len(kernel)
	b = len(kernel[0])
	a2 = int(a/2)
	b2 = int(b/2)
	outAux = np.zeros((Sy, Sx), np.float32)
	
	for i in range(Sy):
		for j in range(Sx):
			g = 0.0
			for s in range(a):
				for t in range(b):
					x = j+t-a2
					y = i+s-b2
					x = min(max(x, 0), Sx-1)
					y = min(max(y, 0), Sy-1)
					g += (kernel[s][t] * img[y][x])
			outAux[i][j] = abs(g)
	
	out = [[255 if x > 255 else x for x in arr] for arr in outAux]
	out = [[0 if x < 0 else x for x in arr] for arr in out]
	
	return np.asarray(out, np.float32)

#Q16
def maskBlur():
	return [[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]

#Q17
def blur(img):
	kernel = maskBlur()
	return convolve(img, kernel)

#Q18
def seSquare3():
	return [[1,1,1],[1,1,1],[1,1,1]]

#Q19
def seCross3():
	return [[0,1,0],[1,1,1],[0,1,0]]

#initG = 256 quando f for min e 0 quando f for max
def morfologicOp(img, structure, initG, f):
	Sx, Sy = size(img)
	numCh = nchannels(img)
	a = len(structure)
	b = len(structure[0])
	a2 = int(a/2)
	b2 = int(b/2)
	if numCh == 1 :
		out = np.zeros((Sy, Sx), np.uint8)
	else :
		out = np.zeros((Sy, Sx, 3), np.uint8)
	
	for i in range(Sy):
		for j in range(Sx):
			if numCh == 1 :
				g = initG
			else :
				g = [initG, initG, initG]
			for s in range(a):
				for t in range(b):
					x = j+t-a2
					y = i+s-b2
					x = min(max(x, 0), Sx-1)
					y = min(max(y, 0), Sy-1)
					if structure[s][t] == 1 :
						if numCh == 1 :
							g = f(img[y][x], g)
						else :
							g = [f(img[y][x][0], g[0]), f(img[y][x][1], g[1]), f(img[y][x][2], g[2])]
			out[i][j] = g
	
	return out
	
#Q20
def erode(img, structure):
	#OBS: Considerando que o centro do elemento estruturante é sempre o centro do retagunlo que ele está inscrito
	return morfologicOp(img, structure, 255, min)

#Q21
def dilate(img, structure):
	a = len(structure)
	b = len(structure[0])
	structAux = [[0 for _ in range(b)] for _ in range(a)]
	for i in range(a):
		for j in range(b):
			structAux[i][j] = structure[a-1-i][b-1-j]
			
	return morfologicOp(img, structure, 0, max)
