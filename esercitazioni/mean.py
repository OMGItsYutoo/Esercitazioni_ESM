import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from time import time

def mean(x,k):
    h=np.ones((k,k))/k**2 #crea una matrice kxk di pesi uniformi 1/k^2
    return ndi.correlate(x,h)

def fast_mean(x,k):
    return ndi.uniform_filter(x,(k,k))

def main():
    plt.close('all')
    x=np.float64(io.imread('images/lena.jpg'))
    tic=time()
    y3=mean(x,3)
    toc=time()
    print(f"Mean correlate: {toc-tic}") #0.0027036666870117188

    tic=time()
    y3_f=fast_mean(x,3)
    toc=time()
    print(f"Mean unif_filter: {toc-tic}") #0.003464221954345703
    
    y5=mean(x,5)
    y10=mean(x,10)
        
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(y3,clim=[0,255],cmap='gray') 
    plt.subplot(2,2,3)
    plt.imshow(y5,clim=[0,255],cmap='gray') 
    plt.subplot(2,2,4)
    plt.imshow(y10,clim=[0,255],cmap='gray') 
    
    # plt.figure(1)
    # plt.subplot(2,2,1)
    # plt.imshow(x,clim=[0,255],cmap='gray')
    # plt.subplot(2,2,2)
    # plt.imshow(y3,clim=None,cmap='gray') 
    # plt.subplot(2,2,3)
    # plt.imshow(y5,clim=None,cmap='gray') 
    # plt.subplot(2,2,4)
    # plt.imshow(y10,clim=None,cmap='gray') 
    
    from skimage.exposure import histogram
    n,b=histogram(x.flatten(), nbins=256)
    plt.figure(3)
    plt.bar(b,n) #plt.plot per l'interpolazione lineare
    plt.axis([0,255,0,1.1*np.max(n)])
    
    n,b=histogram(y3.flatten(), nbins=256)
    plt.figure(4)
    plt.bar(b,n) #plt.plot per l'interpolazione lineare
    plt.axis([0,255,0,1.1*np.max(n)])
    
    n,b=histogram(y10.flatten(), nbins=256)
    plt.figure(5)
    plt.bar(b,n) #plt.plot per l'interpolazione lineare
    plt.axis([0,255,0,1.1*np.max(n)])

    plt.show()

if __name__=='__main__':
    main()