from sklearn.cluster import k_means
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np

def main():
    plt.close('all')
    
    x=np.float64(io.imread('images/fiori256.bmp'))/255
    M,N,L=x.shape
    K=3
    d=np.reshape(x,(M*N,L))
    centroid, idx, sum_var=k_means(d,k)
    y=np.reshape(idx,(M,N))
    

if __name__=='__main__':
    main()
    
    plt.show()
    