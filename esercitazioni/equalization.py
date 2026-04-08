import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from skimage.exposure import equalize_hist,histogram
from skimage.exposure import equalize_adapthist

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def adaptive_equalization(x):
    return equalize_adapthist(x, clip_limit=0.02)

def equalization(x):
    return equalize_hist(x)

def fshs(x):
    m=np.min(x)
    M=np.max(x)
    return 255*(x-m)/(M-m)

def nfun(x):
    x=np.reshape(x,(3,3))
    x=equalize_hist(x)
    return x[1,1]

def loc_equalization(x):    
    return ndi.generic_filter(x,nfun,(3,3))

# def main():
#     plt.close('all')
#     x=np.float64(io.imread('images/marte.jpg'))

#     y=equalization(x)*255
#     y_clahe=adaptive_equalization(x/255)*255    
#     y=fshs(y)
#     y_clahe=fshs(y_clahe)
    
#     n,b=histogram(x,nbins=256)
#     plt.figure(1)
#     plt.bar(b,n)
#     plt.axis([0,255,0,1.1*np.max(n)])
    
#     n,b=histogram(y,nbins=256)
#     plt.figure(2)
#     plt.bar(b,n)
#     plt.axis([0,255,0,1.1*np.max(n)])
    
#     n,b=histogram(y_clahe,nbins=256)
#     plt.figure(3)
#     plt.bar(b,n)
#     plt.axis([0,255,0,1.1*np.max(n)])
    
#     plt.figure(4)
#     plt.subplot(1,3,1)
#     plt.imshow(x,clim=[0,255],cmap='gray')
#     plt.subplot(1,3,2)
#     plt.imshow(y,clim=[0,255],cmap='gray') 
#     plt.subplot(1,3,3)
#     plt.imshow(y_clahe,clim=[0,255],cmap='gray')
    
#     plt.tight_layout()
#     plt.show()

def main():
    plt.close('all')
    x=np.float64(io.imread('images/quadrato.tif'))

    y=equalization(x)*255
    y_clahe=adaptive_equalization(x/255)*255  
    y_loc_eq=loc_equalization(x)*255  
    
    y=fshs(y)
    y_clahe=fshs(y_clahe)
    
    n,b=histogram(x,nbins=256)
    plt.figure(1)
    plt.bar(b,n)
    plt.axis([0,255,0,1.1*np.max(n)])
    
    n,b=histogram(y,nbins=256)
    plt.figure(2)
    plt.bar(b,n)
    plt.axis([0,255,0,1.1*np.max(n)])
    
    n,b=histogram(y_clahe,nbins=256)
    plt.figure(3)
    plt.bar(b,n)
    plt.axis([0,255,0,1.1*np.max(n)])
    
    plt.figure(4)
    plt.subplot(2,2,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(y,clim=[0,255],cmap='gray') 
    plt.subplot(2,2,3)
    plt.imshow(y_clahe,clim=[0,255],cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(y_loc_eq,clim=[0,255],cmap='gray')
    
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    main()