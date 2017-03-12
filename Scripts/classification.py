


## Classification is not good with linear regression, as model changes with adding new points to the set.

## Logistic regression is used instead. A function g(z) is used to 
## Logistic regression:
## g(Theta^T * x)

## g(z) = 1 / ( 1 + e^(-z) )  ::>  Logistic Function. Dist(g(z)) = (0,1). 

## h(x) = 1 / ( 1 + e^(-Theta^T * x)
## g(z) >= 0.5 when z >= 0
## g(z) <= 0.5 when z <= 0

## Substitude z = theta^T*x


## 



def category(input, model,boundary=0.5){
	#comparator
	if input*model>boundary:
		return 1
	else:
		return 0
	
}