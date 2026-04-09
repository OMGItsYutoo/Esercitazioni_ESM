import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from sklearn.cluster import k_means
from skimage.exposure import histogram

def thresholding1(x):
    d=np.reshape(x,(-1,1))    
    centroids,idx,sum_var=k_means(d,2)
    mask=x>np.mean(centroids)
    return mask

def thresholding2(x):
    hist, bin_edges = np.histogram(x, bins=256, range=(0, 255))    
    pk=hist/np.sum(hist)
    Pk=np.cumsum(pk)
    
    intensita=np.arange(256)
    mk=np.cumsum(intensita*pk)

    mg=mk[-1]
    
    mask=(Pk>0) & (Pk<1)
    
    num=(mg*Pk[mask]-mk[mask])**2
    den=Pk[mask]*(1-Pk[mask])

    sigmaB2=num/den
    
    t=np.argmax(sigmaB2)
    
    return x>t
    
def main():
    plt.close('all')
    
    x=np.float64(io.imread("images/cells.png"))
    
    x=x[:,:,0]
    
    y1=thresholding1(x)
    y2=thresholding2(x)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x,cmap='gray',clim=[0,255])
    plt.subplot(1,3,2)
    plt.imshow(y1,cmap='gray',clim=[0,1])
    plt.subplot(1,3,3)
    plt.imshow(y2,cmap='gray',clim=[0,1])
    
if __name__=='__main__':
    main()
    plt.show()