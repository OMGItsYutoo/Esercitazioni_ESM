import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from skimage.exposure import histogram
from skimage.transform import rescale, warp, AffineTransform

def main():
    plt.close('all')
    x=np.float64(io.imread('images/lena.jpg'))

    #y=rescale(x,3,order=0) così si arriva ad avere blocchi 3x3 dello stesso valore di grigio
    y=rescale(x,3,order=1)
    
    A=AffineTransform(translation=(100,50))
    y=warp(x,A,order=1)
    
    n,b=histogram(x,nbins=256)
    plt.figure(1)
    plt.bar(b,n)
    plt.axis([0,255,0,1.1*np.max(n)])
    
    n,b=histogram(y,nbins=256)
    plt.figure(2)
    plt.bar(b,n)
    plt.axis([0,255,0,1.1*np.max(n)])
    
    plt.figure(3)
    plt.subplot(1,2,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(y,clim=[0,255],cmap='gray') 
    
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    main()