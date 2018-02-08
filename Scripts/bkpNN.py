from PIL import Image as im
import numpy as np
from matplotlib import pyplot as pp
from scipy import misc

class bkpnn:
	""" A back propagation neural network """
	layerCount = 0
	shape = None
	weights = []
	
	def __init__(self, layerSize):
		self.layerCount = len(layerSize) - 1
		self.shape = layerSize
		
		self._layerInput = []
		self._layerOutput = []
		
		for (l1,l2) in zip(layerSize[:-1], layerSize[1:]):
			self.weights.append(np.random.normal(scale=0.1,size = (l2,l1+1)))
	
	def TrainEpoch(self, input, target, rate = 0.1):
		
		lnCases = input.shape[0]
		
		self.Run(input)
		# deltas:
		delta = []
		for index in reversed(range(self.layerCount)):
			if index == self.layerCount - 1:
				output_delta = self._layerOutput[index] - target.T
				error = np.sum(output_delta**2)
				delta.append(output_delta * self.sgm(self._layerInput[index], True))
			else:
				delta_pullback = self.weights[index+1].T.dot(delta[-1])
				delta.append(delta_pullback[:-1, :] * self.sgm(self._layerInput[index],True))
		
		for index in range(self.layerCount):
			delta_index = self.layerCount - 1 - index
			
			if index == 0:
				layerOutput = np.vstack([input.T, np.ones([1,lnCases])])
			else:
				layerOutput = np.vstack([self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])
			weightDelta = np.sum(layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0), axis = 0)
			
			self.weights[index] -= rate * weightDelta
		
		return error

	def Run(self, input):
		lnCases = input.shape[0]
		
		self._layerInput = []
		self._layerOutput = []
		for index in range(self.layerCount):
			if index == 0:
				layerInput = self.weights[0].dot(np.vstack([input.T,np.ones([1,lnCases])]))
			else:
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1,lnCases])]))
			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sgm(layerInput))
		return self._layerOutput[-1].T
	
	def sgm(self, x, Der=False):
		if not Der:
			return 1 / (1+np.exp(-x))
		else:
			out = self.sgm(x)
			return out*(1-out)
	
	
#
# if run as a script, create a test script
#

def getY( label, rank=10 ):
	res = np.zeros((1,rank))
	for i in range(rank):
		if str(i) in label:
			res[0,i] = 1
	return res
	
def getRes( output ):
	return output.argmax()

def recognize_digit():
	# The neural net needs 10 nodes at output.
	digitNet = bkpnn( (640, 320, 15, 10) )

	
	fileList = []
	fileRoot = "D:\\MachineLearning\\ML\\Data\\"
	for i in range( 10 ):
		fileList.append( fileRoot + str( i ) + ".png" )

	
	XList = []
	yList = []
	for file in fileList:
		img = im.open( file ).convert("L")
		print(np.asarray(img).shape)
		img = misc.imresize(img,(32,20))
		XList.append( img.reshape( 1,640 ) )
		yList.append( getY( file[-5:] ) )

	X = np.vstack( XList )
	y = np.vstack( yList )
	
	lnMax = 100000
	lnErr = 1e-4
	for i in range(lnMax+1):
		err = digitNet.TrainEpoch(X,y)
		if i % 1000 == 0:
			print("Iteration {0}\t Error: {1:0.6f}".format(i,err))
		if err <= lnErr:
			print("Minimum error reached at iteration {}".format(i))
			break
			
	testFile = "test4.png"
	img = im.open( fileRoot + testFile )
	
	Xtest = misc.imresize(img,(32,20))

	print("input:\n{} \noutput:\n{}".format(testFile[-5],digitNet.Run(Xtest).argmax()))
	
	
	
 
def test():
	bpn = bkpnn((2,2,1))
	print(bpn.shape)
	print(bpn.weights)

	lvInput = np.array([[0,0], [1,1],[0,1],[1,0]])
	lvTarget = np.array([[0.05], [0.05],[0.95],[0.95]])
	lnMax = 100000
	lnErr = 1e-5
	for i in range(lnMax+1):
		err = bpn.TrainEpoch(lvInput,lvTarget)
		if i % 10000 == 0:
			print("Iteration {0}\t Error: {1:0.6f}".format(i,err))
		if err <= lnErr:
			print("Minimum error reached at iteration {}".format(i))
			break
	lvOutput = bpn.Run(lvInput)
	print("input:\n{} \noutput:\n{}".format(lvInput,lvOutput))

if __name__=='__main__':
	recognize_digit()