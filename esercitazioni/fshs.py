import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from skimage.exposure import histogram

def fshs(x):
    m=np.min(x)
    M=np.max(x)
    return 255*(x-m)/(M-m)

def main():
    plt.close('all')
    x=np.float64(io.imread('images/granelli.jpg'))

    y=fshs(x) 
    
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
    #plt.imshow(x,cmap='gray')  in questo modo si effettua il fshs solo in visualizzazione
    plt.subplot(1,2,2)
    plt.imshow(y,clim=[0,255],cmap='gray') 
    
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    main()