

import numpy as np

## Backpropagation:
## delta[:](4) = a[:](4) - y[:]
## delta[:](3) = T( theta[:,3] * delta[:](4) .* g'( z( 3 ) )


class Neural_Network( object ):
	StaticList = []
	def __init__(self, inputSize = 2, layerSizes = [ 5 ], outputSize = 1 ):
		self.inputSize 			= inputSize
		self.outputSize 		= outputSize
		
		self.layerSizes 		= np.asarray( layerSizes )
		if type(layerSizes) is int:
			self.layers 		= 1
		else:
			self.layers				= len( self.layerSizes )
		

		self.W1 = np.random.randn( self.inputSize, self.layerSizes[0] ) 
		self.W2 = np.random.randn( self.layerSizes[0] , self.outputSize ) 
		
	
	def forward(self,  X ):		
		self.z2 = np.dot( X, self.W1 )
		self.a2 = self.sigmoid( self.z2 )
		self.z3 = np.dot( self.a2, self.W2 )
		yHat 	= self.sigmoid( self.z3 )
		return yHat
		
	def sigmoid(self, z ):
		return 1 / ( 1 + np.exp( -z ) )
	def sigmoidPrime(self, z ):
		return np.exp( -z ) / ( ( 1 + np.exp( -z ) )**2 )
		
	def costFunctionPrime(self, X, y ):
		self.yHat = self.forward( X )
		
		delta3 = np.multiply( -( y - self.yHat), self.sigmoidPrime( self.z3 ) )
		dJdW2 = np.dot( self.a2.T, delta3 )
		
		delta2 = np.multiply( delta3, self.W2.T ) * self.sigmoidPrime( self.z2 )
		dJdW1 = np.dot( X.T, delta2 )
		
		# returns a input x hidden and a hidden x output array
		return dJdW1, dJdW2
		
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
			( dJdW1, dJdW2 ) = self.costFunctionPrime( X, y )
			temp1 = np.zeros( np.shape( self.W1 ) )
			temp2 = np.zeros( np.shape( self.W2 ) )
			
			temp1 = self.W1 - self.k * dJdW1
			temp2 = self.W2 - self.k * dJdW2
			
			if self.converged(temp1, self.W1, self.minError ) and self.converged(temp2, self.W2, self.minError ):
				converging = False
			
			counter += 1
			
			self.W1 = temp1
			self.W2 = temp2
			
		print("Regression finished after {} iterations.".format( str( counter ) ) )
		
		
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

def getY( label, rank=10 ):
	res = np.zeros((rank,1))
	for i in range(rank):
		if str(i) in label:
			res[i,0] = 1
	return res
	
	
def recognize_digit():
	# The neural net needs 10 nodes at output. Y is the either
	digitNet = Neural_Network( 4000, [25, 10], 10 )
	
	
	
	
	
	
	
def main():
	recognize_digit()
	



if __name__=='__main__':
	main()