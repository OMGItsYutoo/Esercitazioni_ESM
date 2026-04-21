import skimage.data as data
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from skimage.feature import hog
from sklearn.cluster import k_means

def main():
    x=np.float64(io.imread("images/textures.png"))
    
    y=hog(x, feature_vector=False)
    k = 4
    M = y.shape[0]
    N = y.shape[1]
    y = np.reshape(y, (M*N,-1))
    centroid, idx, sum_var = k_means(y, k)
    idx = np.reshape(idx, (M,N))

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x, clim=[0,255], cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(idx, clim=[0,k-1])
    plt.show()

if __name__=='__main__':
    main()
    plt.show()