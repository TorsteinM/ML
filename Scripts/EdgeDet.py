from scipy import misc
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from skimage import feature



f = misc.imread('test.png')

print(f.shape)
#f+=2.5-5*np.random.randn(*f.shape)

edge1=feature.canny(f[:,:,3])
edge2=feature.canny(f[:,:,3],sigma=0.5)
#edges[:,:,1] = 255 * feature.canny(f[:,:,1])
#edges[:,:,2] = 255 * feature.canny(f[:,:,2])


U,s,V=la.svd(f[:,:,3],full_matrices=False)

print(s)
red=np.zeros(s.shape)
red[:9] = s[:9]
print(red)
S=np.diag(red)
recon = np.dot(U,np.dot(S,V))
plt.subplot(211)
plt.imshow(f[:,:,3], cmap=plt.cm.gray)
plt.subplot(212)
plt.imshow(recon, cmap=plt.cm.gray)
#plt.subplot(313)
#plt.imshow(np.real(np.fft.ifft2(dftabs)), cmap=plt.cm.gray)

plt.show()





