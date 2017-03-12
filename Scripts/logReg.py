

## Cost function
import inspect
from math import exp
from matplotlib import pyplot as pp
import numpy as np
from numpy import transpose as T
from math import log


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
##

##

def convergeVec(a,b,minErr):
	if sum(abs(a-b)) < minErr:
		return True
	else:
		return False

	
def	gradDesc( theta, trSet, alpha = 0.003, minErr = 1/10**5 ):
	
	k=alpha/len(trSet)
	#print(len(trSet))
	count=0
	converged = convergeVec
	converging = True
	while converging:
		temp = np.zeros( np.shape( theta ) )
		
		#p=np.concatenate((theta,[-1]),axis=0)
		
		inner = np.reciprocal( 1 + np.exp( - np.sum( theta * trSet[:,:-1], axis = 1 ) ) ) - trSet[:,-1]
	#	print(np.shape(inner))
		outer = inner * T(trSet[:,:-1])
	#	print(np.shape(outer))
		delta = np.sum(outer,axis=1)
	#	print(delta)
		#print(inner[:30])
		
		
		temp = theta - k * delta
		if converged(temp, theta, minErr):
			converging = False
		count+=1
		theta = temp
	
	print("Logistic regression converged after {} iterations.".format(count))
	
	return theta
	

def main():

	sizes=[]
	diags=[]
	for line in open("data.csv").readlines():
		values = line.split(',')
		if len(values)>3:
			sizes.append(values[2])
			if values[1]=='M':
				diags.append(1.)
			else:
				diags.append(0.)
	
	#for i in range(1,31):
	#	print("Size {} was diagnosed as {}".format(sizes[i],diags[i]))
	
	X    = np.asarray( sizes[1:], dtype=float )
	Y    = np.asarray( diags[1:], dtype=float )
	clm1 = np.ones( np.shape( X ), dtype=float )
	
	trSet = T( np.vstack( ( clm1, X, Y ) ) )
	initTheta = np.asarray([ 0.1, 0.7 ])			# model for the boundary between benign and malign
	
	theta = gradDesc( initTheta, trSet )
	
	
	print(theta)
	
	Xarr = np.arange(5.,30.,0.5)
	Yarr = np.asarray(list(map(lambda x:theta[0] + x*theta[1] , Xarr)))
	
	pp.scatter(X,Y)
	pp.plot(Xarr,Yarr,c='r')
	pp.show()
	

if __name__=='__main__':
	main()
