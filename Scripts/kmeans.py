# A simple kmeans algorithm. Very unfinished. Needs work to take n-dimensions and arbitrary input data.
# Initial centroids are chosen by generating random numbers within the sample distribution without any weighting.
# K-means++ might be a nice way to implement better initialization.
#
#
#
import numpy as np
from random import randint as rn
from numpy import zeros as zs
from matplotlib import pyplot as pp

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
	minErr=0.001
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
		print("The centroids after " + str(passes) + " passes: ")
		print(centr)
		passes+=1
		res=centrPass(x,y,centr)
		for i in range(ncl):
			if(abs(centr[i][0]-res[i][0]) + abs(centr[i][0]-res[i][0]) > minErr ):
				running=True
				if(res[i][0] == 0 and res[i][1] == 0):
					centr[i][0]=rn(min(x),max(x))
					centr[i][1]=rn(min(y),max(y))
				else:
					centr[i][0]=res[i][0]
					centr[i][1]=res[i][1]
	print("The centroids after " + str(passes) + " passes: ")
	print(centr)				
	return centr
		
		
	


def main(n=50,minVal=0,maxVal=9):
	x=zs(n)
	y=zs(n)
	for i in range(n):
		x[i]=rn(minVal,maxVal)
		y[i]=rn(minVal,maxVal)
	
	centr=kmeans(x,y,)
	c1=centr[:,0]
	c2=centr[:,1]
	
	
	pp.scatter(x,y)
	pp.scatter(c1,c2,c='r',s=100)
	pp.show()

if __name__=="__main__":

	main()
	
	