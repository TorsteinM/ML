

import numpy as np
from matplotlib import image as im

## Backpropagation:
## delta[:](4) = a[:](4) - y[:]
## delta[:](3) = T( theta[:,3] * delta[:](4) .* g'( z( 3 ) )


class Neural_Network( object ):
	def __init__(self, inputSize = 2, layerSizes = [ 5 ], outputSize = 1 ):
		self.W1 = np.random.randn( inputSize, layerSizes[0] )
		self.W2 = np.random.randn( layerSizes[0] , layerSizes[1] )
		self.W3 = np.random.randn( layerSizes[1] , outputSize )
		
	
	
	# forward estimates y given input X and should work with two dimensional arrays with arbitrary sample items
	def forward(self,  X ):	
		# 1st hidden layer, input weights(X->z2:W1) and activation(z2->a2)
		self.z2 = np.dot( X, self.W1 )
		self.a2 = self.sigmoid( self.z2 )
		
		# 2nd hidden layer, input weights(a2->z3:W2) and activation(z3->a3)
		self.z3 = np.dot( self.a2, self.W2 )
		self.a3 = self.sigmoid( self.z3)
		
		# output layer, input weights(a3->z4:W3) and activation (z4->yHat)
		self.z4 = np.dot( self.a3, self.W3 )
		yHat 	= self.sigmoid( self.z4 )
		return yHat
		
	def sigmoid(self, z ):
		return 1 / ( 1 + np.exp( -z ) )
		
	def sigmoidPrime(self, z ):
		return np.exp( -z ) / ( ( 1 + np.exp( -z ) )**2 )
		
		
		#function that returns the derivatives for all the W matrices
	def costFunctionPrime(self, X, y ):
		self.yHat = self.forward( X )
		
		#W3
		print(np.shape(self.yHat))
		print(np.shape(self.z4))
		print(np.shape(self.a3))
		input("waiting...")
		
		delta4 = np.multiply( -( y - self.yHat), self.sigmoidPrime( self.z4 ) )
		dJdW3  = np.dot( self.a3.T, delta4 )
		
		print(np.shape(dJdW3))
		print(np.shape(delta4))
		print(np.shape(self.W3.T))
		print(np.shape(self.z3))
		input("waiting...")
		#W2
		d4w3 = np.multiply( delta4, self.W3.T )
		delta3 = d4w3 * self.sigmoidPrime( self.z3 )
		dJdW2  = np.dot( self.a2.T, delta3 )
		
		#W1
		print(np.shape(delta3))
		print(np.shape(self.W2.T))
		print(np.shape(self.z2))
		input("waiting..")
		delta2 = np.multiply( delta3, self.W2.T ) * self.sigmoidPrime( self.z2 )
		dJdW1  = np.dot( X.T, delta2 )
		
		# returns a input x hidden and a hidden x output array
		return dJdW1, dJdW2, dJdW3
		
	def converged(self, a, b, minErr):
		if np.sum( np.abs( a - b ) ) < minErr:
			return True
		return False
	
	
	def train(self, X, y, alpha = 0.003, minErr = 1/10**5 ):
		
		self.minError = minErr
		self.k = alpha / max( np.shape(y) )
		converging = True
		counter = 0
		
		while converging:
			( dJdW1, dJdW2, dJdW3 ) = self.costFunctionPrime( X, y )
			temp1 = np.zeros( np.shape( self.W1 ) )
			temp2 = np.zeros( np.shape( self.W2 ) )
			temp3 = np.zeros( np.shape( self.W2 ) )
			
			temp1 = self.W1 - self.k * dJdW1
			temp2 = self.W2 - self.k * dJdW2
			temp3 = self.W2 - self.k * dJdW2
			
			if self.converged( temp1, self.W1, self.minError ) and self.converged( temp2, self.W2, self.minError ) and self.converged( temp3, self.W3, self.minError ):
				converging = False
	
			counter += 1
			self.W1 = temp1
			self.W2 = temp2
			self.W2 = temp3
			
		print("Regression finished after {} iterations.".format( str( counter ) ) )
		

def getY( label, rank=10 ):
	res = np.zeros((1,rank))
	for i in range(rank):
		if str(i) in label:
			res[0,i] = 1
	return res
	
	
def recognize_digit():
	# The neural net needs 10 nodes at output. Y is the either
	digitNet = Neural_Network( 4000, [100, 25], 10 )

	
	fileList = []
	fileRoot = "D:\\MachineLearning\\ML\\Data\\"
	fileList.append(fileRoot + "test7.png")
	for i in range( 10 ):
		fileList.append( fileRoot + str( i ) + ".png" )

	
	XList = []
	yList = []
	for file in fileList:
		img = im.imread( file )
		XList.append( img[:,:,3].reshape( 1,4000 ) )
		yList.append( getY( file[-5:] ) )
	
	X = np.vstack( XList )
	y = np.vstack( yList )
	
	digitNet.train( X, y )
	
	
	
	
	
	
	
def main():
	recognize_digit()

def test():
	X = np.asarray([ [3,5], [5,1], [10,2] ], dtype = float )
	y = np.asarray([ [75], [82], [93] ], dtype = float )
	
	yNorm = y.max()
	xNorm = X.max()
	
	X = X / xNorm
	y = y / yNorm			## Bring X and y values into the normalized space.
	
	NN = Neural_Network( 2, [3], 1 )
	NN.train( X, y, 0.003, 1/10**5 )
	
	print( NN.W1 )
	print( NN.W2 )
	
	print( NN.forward( [8,4] / xNorm ) * yNorm )		#Have to bring the input into the normalized space, and the restore the result after by multiplying with the y normalization.

if __name__=='__main__':
	main()