import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from time import time

def var(x,k):
    return ndi.generic_filter(x,np.var,(k,k))

def fast_var(x,k):
    mean_of_squares = ndi.uniform_filter(x**2, (k,k))
    square_of_mean=ndi.uniform_filter(x,(k,k))**2
    return mean_of_squares-square_of_mean
    
def main():
    plt.close('all')
    x=np.float64(io.imread('images/filamento.jpg'))

    tic=time()
    y3=var(x,3)
    toc=time()
    print(f"Var gen_filter: {toc-tic}") #2.2463674545288086
    
    tic=time()
    y3=fast_var(x,3)
    toc=time()
    print(f"Var unif_filter: {toc-tic}") #0.010646581649780273

    y5=fast_var(x,5)
    y10=fast_var(x,10)
        
    # plt.figure(1)
    # plt.subplot(2,2,1)
    # plt.imshow(x,clim=[0,255],cmap='gray')
    # plt.subplot(2,2,2)
    # plt.imshow(y3,clim=[0,255],cmap='gray') 
    # plt.subplot(2,2,3)
    # plt.imshow(y5,clim=[0,255],cmap='gray') 
    # plt.subplot(2,2,4)
    # plt.imshow(y10,clim=[0,255],cmap='gray') 
    
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(y3,clim=None,cmap='gray') 
    plt.subplot(2,2,3)
    plt.imshow(y5,clim=None,cmap='gray') 
    plt.subplot(2,2,4)
    plt.imshow(y10,clim=None,cmap='gray') 
    
    from skimage.exposure import histogram
    n,b=histogram(x.flatten(), nbins=256)
    plt.figure(3)
    plt.bar(b,n) #plt.plot per l'interpolazione lineare
    plt.axis([0,255,0,1.1*np.max(n)])
    
    # n,b=histogram(y3.flatten(), nbins=256)
    # plt.figure(4)
    # plt.bar(b,n) #plt.plot per l'interpolazione lineare
    # plt.axis([None,None,0,1.1*np.max(n)])
    
    # n,b=histogram(y10.flatten(), nbins=256)
    # plt.figure(5)
    # plt.bar(b,n) #plt.plot per l'interpolazione lineare
    # plt.axis([None,None,0,1.1*np.max(n)])

    plt.show()

if __name__=='__main__':
    main()