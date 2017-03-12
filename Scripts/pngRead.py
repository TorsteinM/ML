import numpy as np
import png
from matplotlib import pyplot as pp 
from random import randint as rn
from numpy import zeros as zs
import sys


def centrPass(x,y,centr,ndim=2,ncl=5):
	numUpdates=0
	minErr=0.01
	res=zs((ncl,ndim+1))
	for i in range(len(x)):
		closest=0
		best=abs(x[i]-centr[0][0]) + abs(y[i]-centr[0][1])
		for j in range(1,ncl):
			dist=abs(x[i]-centr[j][0]) + abs(y[i]-centr[j][1])
			if(dist<best):
				best=dist
				closest=j
		res[closest][0]+=x[i]
		res[closest][1]+=y[i]
		res[closest][ndim]+=1
	
	for i in range(ncl):
		if(res[i][ndim]!=0):
			res[i][0]=res[i][0]/res[i][ndim]
			res[i][1]=res[i][1]/res[i][ndim]
	
	return res


def kmeans(x,y,ndim=2,ncl=5):
	#init the number of centroids and initialize them to random points.
	centr=zs((ncl,ndim))
	minErr=0.01
	#list of [x,y] pairs.
	for i in range(ncl):
		centr[i][0]=rn(min(x),max(x))
		centr[i][1]=rn(min(y),max(y))
	#centr now contains ncl item list of [x,y] pairs.
	numUpdates=1
	passes=0
	running=True
	while running:
		running=False
		#print("The centroids after " + str(passes) + " passes: ")
		#print(centr)
		passes+=1
		res=centrPass(x,y,centr,2,ncl)
		for i in range(ncl):
			if(abs(centr[i][0]-res[i][0]) + abs(centr[i][0]-res[i][0]) > minErr ):
				running=True
				if(res[i][0] == 0 and res[i][1] == 0):
					centr[i][0]=rn(min(x),max(x))
					centr[i][1]=rn(min(y),max(y))
				else:
					centr[i][0]=res[i][0]
					centr[i][1]=res[i][1]
	#print("The centroids after " + str(passes) + " passes: ")
	#print(centr)				
	return centr


def processImage(imagePath="7.png",ncl=6):
	#load image as binary
	f = open(imagePath,'rb')
	r = png.Reader(f)
	width,height,imgData,_ = r.asDirect()
	#print("width: " + str(width) + " height: " + str(height) + " pixels: " + str(pixels))
	
	#treat image data iterator as list
	l=list(imgData)
	
	#initialize result arrays for making (x,y) scatter
	x=[]
	y=[]
	for row in range(height):
		for pixel in range(width):
			index=pixel*4 + 3
			if (l[row][index]>0):
				y.append(height-row)
				x.append(pixel)
	

	#print(len(x))
	#print(len(y))
	# making 5 clusters with the 2 dimensions of the (x,y) scatter.
	cl=kmeans(x,y,2,ncl)
	if __name__=='__main__':
		pp.scatter(x,y)
	
		pp.scatter(cl[:,0],cl[:,1],s=200,c='r')
	
		pp.show()
	else:
		
		sortCl=zs(cl.shape)
		indices=np.argsort(cl[:,0]*cl[:,0]+cl[:,1]*cl[:,1])
		for i in range(len(indices)):
			sortCl[i,:]=cl[indices[i],:]
			
		return sortCl

if __name__=='__main__':
	processImage()

	


