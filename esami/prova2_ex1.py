import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io 

def patch_fun(x,r):
    xc=x[x.shape[0]//2] #il blocco viene vettorizzato 
    y=x[np.abs(x-xc)<=r] #np.abs(x-xc)>=r rappresenta una maschera vettoriale, true dove il pixel è nel range [xc-r,xc+r] (calcolo matematico con valore assoluto)
    if len(y)>4:
        v=np.mean(x)
    else:
        v=np.mean(y)
    return v

def filtro_sigma(x,k,sigma):
    y=ndi.generic_filter(x,patch_fun, (k,k), extra_arguments=(2*sigma,))
    return y

def main():
    plt.close('all')
    
    x=np.float64(io.imread("images/barbara.png"))
    M,N=x.shape
    
    d=20
    n=d*np.random.randn(M,N)    
    noisy=x+n
    
    y=filtro_sigma(noisy,7,2)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(noisy,clim=[0,255],cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(y,clim=[0,255],cmap='gray')

if __name__=='__main__':
    main()
    plt.show()
