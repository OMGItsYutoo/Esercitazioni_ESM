import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from skimage.exposure import histogram
from bitop import bitset, bitget

def main():
    plt.close('all')
    x=np.fromfile('images/lena.y', np.uint8)
    x=np.reshape(x, (512,512))
    m=np.fromfile('images/marchio.y', np.uint8)
    m=np.reshape(m, (350,350))
    m=m.T    
    
    plt.figure(1)
    plt.imshow(m, clim=[0,1], cmap='gray')
    
    x=x[50:400,50:400] #ridimensionamento dell'immagine principale, si sarebbe comunque potuto utilizzare del padding
    bit_plane=6
    y=bitset(x,bit_plane,m)
    b=bitget(x,bit_plane)
    
    plt.figure(2)
    plt.subplot(1,2,1)
    plt.imshow(y, clim=[0,255], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(m, clim=[0,1], cmap='gray')
    
    n,b=histogram(x,nbins=256)
    plt.figure(3)
    plt.bar(b,n)
    plt.axis([0,255,0,1.1*np.max(n)])
    
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    main()