#
# Imports
#

from sklearn.neural_network import MLPClassifier
from PIL import Image as im
import numpy as np
from scipy import misc
import glob

def getY( label, rank=10 ):
	res = np.zeros((1,rank))
	for i in range(rank):
		if str(i) in label:
			res[0,i] = 1
	return res

def main():
	clf = MLPClassifier(hidden_layer_sizes=(640,32,10))
	
	fileList = []
	fileRoot = "D:\\MachineLearning\\ML\\Data\\"
	for i in range( 10 ):
		fileList.append( fileRoot + str( i ) + ".png" )
		
	XList = []
	yList = []
	for file in fileList:
		img = im.open( file )
		img = misc.imresize(img,(32,20))
		print(img.shape)
		XList.append( img[:,:,3].reshape( 1,640 ) )
		yList.append( getY( file[-5:] ) )
	
	X = np.vstack( XList )
	y = np.vstack( yList )
	
	print(np.sum(X))
	
	clf.learning_rate_init=0.003
	clf.max_iter=100000
	clf.tol=1e-8
	#clf.alpha=1
	clf.fit(X,y)

	testFile = "test4.png"
	img = im.open( fileRoot + testFile )
	Xtest = misc.imresize(img,(32,20))[:,:,3].reshape( 1,640 )
	
	print("input: {}  output: {}".format(testFile[-5],clf.predict(Xtest)))

if __name__=='__main__':
	main()


