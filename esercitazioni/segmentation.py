from sklearn.cluster import k_means
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np

def main():
    plt.close('all')
    
    x=np.float64(io.imread('images/Fiori256.bmp'))/255
    M,N,L=x.shape
    
    K=3
    d=np.reshape(x,(-1,L))
    centroid, idx, sum_var=k_means(d,K)
    y=np.reshape(idx,(M,N))
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x)
    plt.subplot(1,2,2)
    plt.imshow(y,clim=[0,K-1])

if __name__=='__main__':
    main()
    
    plt.show()
    