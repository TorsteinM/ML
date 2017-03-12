
from math import exp
from matplotlib import pyplot as pp
import numpy as np
from numpy import transpose as T

## x0 == 1, x1 -> xN in [0,1]

## Cost:: J(Ø)lin = 1/m * sum( 1/2 * ( hØ(xi) - yi ) , i in range(1,m) )

## Cost:: J(Ø)log = 1/m * sum( cost( hØ(xi) - yi ) )

##		cost =:	"half of squared error"

## Original function is nonconvex for sigmoid(?)

## Cost (hØ(x),y) = -log(hØ(x)) if y=1
##				  = -log(1 - hØ(x)) if y=0

## simplified cost function and gradient descent for logistic regression.

## Compressed cost function:
## Cost (hØ(x),y) = -y * log ( hØ(x) ) - ( 1 - y ) * log( 1 - hØ(x) )

## ( Maximum Likelihood Estimation )
## J(Ø) = - 1/m * sum( y * log( hØ(xi) ) + (1 - yi) * log(1 - hØ(xi)) )


## To find parameters Ø
	

def logreg( theta, X ):
	N = pSize( len(theta), len(X), logreg )
	res = 0
	for i in range(N):
		res += theta[i] * X[i]

	return 1 / ( 1 + exp( -res ) )

	## Convergence check for ints.
def convInt(a,b,minErr):
	if abs(a-b) < minErr:
		return True
	return False
	
	## Convergence check for numpy arrays.
def convVec(a,b,minErr):			
	if sum(abs(a-b)) < minErr:
		return True
	return False

	## Convergence check for lists.
def convList(a,b,minErr):
	sum=0
	for i in range(len(a)):
		sum+=abs(a[i]-b[i])
	if sum < minErr:
		return True
	return False
	

def gradDesc( theta, TrainingSet, alpha=0.003, minDelta=1/10**10 ):
	##currently only implemented for linear regression.
	##traininset is 2 dimensional array where the rows are as follows: [X0,X1,X2...Xn,Y]. 
	##theta is the [P0,P1,P2...Pn] hypothesis.
	##if convergence fails, try lower alpha value.
	
	converging = True
	k = alpha / np.shape(TrainingSet)[0]
	
	converged = convVec		# currently implemented only for numpy arrays.
	
	count = 0
	
	while converging:
		temp=np.zeros(np.shape(theta))
		
		p=np.concatenate( ( theta, [-1] ), axis=0 ) # append -1 to theta vector for '-Y', which is the last column in the Set
	#	print(p)
		inner = np.sum( TrainingSet*p, 1 ) # Calculates the model mismatch
	#	print(inner)
		outer = inner * T( TrainingSet[:,:-1] ) # Multiply with X-vector to get gradient
	#	print (outer)
		delta = np.sum( outer, axis = 0 )		# Sums all gradients to gradients*m
	#	print(delta)
	#	input("Press Enter to continue...")
	
		temp = theta - k * delta	#subtracts a alpha portion of the mean gradient of the error(k=alpha/m)
		
		if converged(temp,theta,minDelta):	# calls the designated function to check for convergence. 
			converging = False
		theta = temp
		count += 1
	
	print("Gradient descent finished after {} iterations.".format(count))
	return theta

def main():
    # param(x0,x1,x1^2,y0) => theta (p0,p1,p2), try to find 3+4x+x**2
	trSet = np.asarray([[1,3,9,24],[1,5,25,48],[1,-7,49,24]])

	initTheta = np.asarray([55,55,55])
	
	theta = gradDesc( initTheta, trSet )
	print(theta)
	

if __name__=='__main__':
	main()
