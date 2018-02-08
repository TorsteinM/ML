

## 				0  ::<	0	::<					theta(x) element in R^4, want:
##		0 ::<	0  ::<	0	::<		0			[1;0;0;0] for case 1
##		0 ::<	0  ::<	0	::<		0	::> 	[0;1;0;0] for case 2
##		0 ::<	0  ::<	0	::<		0			[0;0;1;0] for case 3
##				0  ::<	0	::<		0			[0;0;0;1] for case 4

# Training set: (x1,y1),(x2,y2) ...(xm,ym), where yi =: 1 of 4 cases.

# L 	= total number of layers
# sl 	= no. of units in each layer.
# y 	element in R^K; for K classes.
# Unit 	= A neuron/point/node in the network.

# layers = [ len(x), len(layer1), len(layerN), ..., len(y) ]
# Forward function:


class Neural_Network( object ):
	def __init__(self):
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3
	
	self.W1 = np.random.randn( self.inputLayerSize, self.hiddenLayerSize )
	self.W2 = np.random.randn( self.hiddenLayerSize, self.outputLayerSize )

	def forward( self, X ):
		self.z2 = np.dot( X, self.W1 )
		self.a2 = self.sigmoid( self.z2 )
		self.z3 = np.dot( self.a2, self.W2 )
		yHat = self.sigmoid( self.z3 )
		return yHat
		
	def sigmoid( z ):
		return 1 / ( 1 + np.exp( -z ) )
		
	def sigmoidPrime( z ):
		return np.exp( -z ) / ( ( 1 + np.exp( -z ) )**2 )
	def costFunctionPrime( self, X, y ):
		self.yHat = forward( X )
		
		delta3 = np.multiply( - ( y - self.yHat ), self.sigmoidPrime( self.z3 ) )
		dJdW2 = np.dot( self.a2.T, delta3 )
		
		delta2 = np.dot( delta3, self.W2.T ) * self.sigmoidPrime( self.z2 )
		dJdW1 = np.dot( X.T, delta2 )
		
		return dJdW1, dJdW2

# Cost function

def myRegul( theta, addCost = 100 ):
	eye = np.identity( np.shape( theta )[ 1 ] )
	eye[ 0, 0 ] = 0					# bias to first element
	return addCost * eye * theta



def UnitLogLikelihood( theta, x, y):
	return y * np.log( mySigmoid( theta * x ) )  + ( 1 - y ) * np.log( 1 - mySigmoid( theta * x ) ) + myRegul( theta )

	
def meanJ( theta, x, y ):
	# mean cost for trainingset for the network with the given hypothesis "Theta":
	# cost is low likelihood given the simplified log likehood function.
	# -1 / m * sum( NetworkCost( theta , x, y ), for all samples )
	# NetworkCost = sum ( UnitCost( theta, x, y ), for all units ) + Regularization cost
	# 
	cost = -1 / m * sum( sum( UnitLogLikelihood( theta, x, y ) , range( 1, K ) ), range( 0, m ) ) + Reg( theta )

def gradOfMeanJ( theta, x, y ):
	# to fit theta to maximum log likelihood, one can use gradient descent. But then we need the gradient. How to find it:
	# Backpropagation to find the derivative of the mean cost of the network for a given dataset.
	# 
	backPropagation( theta x, y )
	
	def converged( a, b, minErr ):
		if sum( abs( a - b ) ) < minErr:
			return True
		else:
			return False
	
def bestFit( theta, x, y, alpha=0.001, minErr = 1/10**5):
	converging = True
	k = -1 * alpha / len(y)
	while converging:
		delta = backPropagation( theta, x, y ) + myRegul( theta, mY )
		temp = theta - k * delta
		if converged( temp, theta, minErr):
			converging = False
		theta = temp
	return theta
	
	
	
	
	
	
	
	
def main():
	NN = Neural_Network()
	

	
	
if __name__ = '__main__':
	main()
