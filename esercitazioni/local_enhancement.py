import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.exposure import histogram

def fast_mean(x,k):
    return ndi.uniform_filter(x,(k,k))

def fast_std(x,k):
    mean_of_squares=ndi.uniform_filter(x**2,(k,k))
    square_of_means=ndi.uniform_filter(x,(k,k))**2
    return np.sqrt(mean_of_squares-square_of_means)
    
def local_enhancement(x,k):
    mean=np.mean(x)
    std=np.std(x)
    MEAN=fast_mean(x,k)
    STD=fast_std(x,k)
    mask=(MEAN<=0.4*mean) & (STD>=0.02*std) & (STD<=0.4*std) #bassa luminosità e contrasto nel range 2-40%
    return x+3*x*mask #E=4

def fshs(x):
    m=np.min(x)
    M=np.max(x)
    return 255*(x-m)/(M-m)

def main():
    plt.close('all')
    x=np.float64(io.imread('images/filamento.jpg'))

    y=local_enhancement(x,3)
    
    n,b=histogram(x,nbins=256)
    plt.figure(1)
    plt.bar(b,n)
    plt.axis([0,255,0,1.1*np.max(n)])
    
    n,b=histogram(y,nbins=256)
    plt.figure(2)
    plt.bar(b,n)
    plt.axis([0,255,0,1.1*np.max(n)])
    
    plt.figure(3)
    plt.subplot(1,3,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(y,clim=[0,255],cmap='gray') 
    
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    main()