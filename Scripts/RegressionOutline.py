
## Warning: Code is written mostly to line out the theory. ##

# x^sup(i),y^sup(i) : ith training example.

#	  [Training SET]
#			:::
# 	      	 :
# 	[Learning Algorithm]
#			:::
# 	   		 :
# in ::> [H(ypothesis)] ::> Estimate   :: Linear Reg h(x) = param0 + param1*x

## linear regression
## Cost funtion for Minimum Squared Error 

## Feature scaling:
## Features can be scaled to -1 > x 1 and normalize mean to 0 for quicker convergence.

## Learning Rate alpha:
## Experiment to get good/fast convergence.

## Exact solution(normal equation):
## theta =(X.t * X)^-1 X.t*y

## J(Theta)
def Cost( params, trainingSet ) {
	cost=0
	m=len(trainingSet)
	for i in trainingSet:
		cost += (1/2m) * (params - i)**2
	#end
	return cost
}


## (Batch) Gradiant descent
## Susceptible to finding local maxima
## 
def gradDesc(params, trainingSet, alpha=0.5){
	# d/dparams(1/2m * sum((params(x)-y)**2)) ::>
	# params[0]: the constant
	# params[1]: first order variable
	# alpha: learning speed
	
	# (hparam(x)) =~ y 
	res = params
	k = alpha/len(trainingSet)
	for i in range(len(params)):
		res[i] = params[i] - k * sum(map(lambda x: (params*x[:len(params-1)) - x[-1:] * x[i],trainingSet))

	
	
	print("Old Params: %3.2f by %3.2f" % params)
	print("Result:     %3.2f by %3.2f" % res)
	return res
}

## Goal: min(J
def main(){
	paramSpace = []
	trainingSet = []
	minCost = float('inf')
	minParams = (0,0)
	for params in paramSpace:
		result = Cost(params,trainingSet) #J(params)
		
		if result < minCost:			# implement gradient descent?
			minCost=result			#
}
if __name__=='__main__':
	main()