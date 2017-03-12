
import pngRead
from matplotlib import pyplot as pp
from numpy import zeros as zs
import numpy as np
from math import sqrt

	
def compareProps( a , b ):
	cost = 0
	for i in range( len( a[:,0] ) ):
		# MSE
		cost+=sqrt( ( a[i,0] - b[i,0] )**2 + (a[i,1] - b[i,1] ) **2 )
	return cost

def main():

	# Populating property vectors for each number.
	properties=[] # list of 5x2 numpy.ndarray property vectors.
	for i in range(10):
		imgString = str(i) + ".png"	
		properties.append( pngRead.processImage( imgString ) )
	
	# Reading test image/number
	testString = "test.png"
	test = pngRead.processImage( testString )
	
	minCost = 16**2
	minCostInd = -1
	# MMSE for test image:
	for i in range( len( properties ) ):
		cost = compareProps( test, properties[i] )
		print( cost )
		if cost < minCost:
			minCost = cost
			minCostInd = i
	print("I guess the number you wrote was " + str(minCostInd) + ".")
		
	
if __name__ == '__main__':
	main()