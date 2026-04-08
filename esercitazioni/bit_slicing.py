import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from bitop import bitget,bitset
from skimage.exposure import histogram

def main():
    plt.close('all')
    x=io.imread('images/frattale.jpg') #non ha senso la coversione in float

    plt.figure(1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    
    plt.figure(2)
    for i in range(8):
        y=bitget(x,i)
        plt.subplot(2,4,1+i)
        plt.imshow(y,clim=[0,1],cmap='gray') #matrice booleana 
        plt.title(f"Bit plane {i}")
        
    plt.tight_layout()
    
    y=np.copy(x)
    plt.figure(3)
    for i in range(8):
        y=bitset(y,i,0)
        plt.subplot(2,4,1+i)
        plt.imshow(y,clim=[0,255],cmap='gray') #matrice booleana 
        plt.title(f"Bit plane {i} a zero")
            
    n,b=histogram(x,nbins=256)
    plt.figure(4)
    plt.bar(b,n)
    plt.axis([0,255,0,1.1*np.max(n)])
    
    plt.show()

if __name__=='__main__':
    main()