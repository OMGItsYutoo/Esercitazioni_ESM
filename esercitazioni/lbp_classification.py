import skimage.data as data
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from skimage.feature import local_binary_pattern

def get_feature(x):
    y=local_binary_pattern(x,P=8,R=1,method='ror')
    h,b=np.histogram(y.flatten(),np.arange(257),density=True)
    return h

def main():
    x1=np.float64(io.imread("images/brick.png"))    
    x2=np.float64(io.imread("images/grass.png"))    
    x3=np.float64(io.imread("images/gravel.png"))    
    
    h1=get_feature(x1)
    h2=get_feature(x2)
    h3=get_feature(x3)
    
    x=np.float64(io.imread("images/img2.jpg"))    
    h=get_feature(x)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x,clim=[0,255],cmap="gray")
    plt.subplot(1,2,2)
    plt.bar(np.arange(256),h)
    
    d1=np.sum(np.abs(h-h1))
    d2=np.sum(np.abs(h-h2))
    d3=np.sum(np.abs(h-h3))
    
    if (d1<d2) & (d1<d3):
        plt.title("L'immagine è un muro")
    elif d2<d3:
        plt.title("L'immagine è un prato")
    else:
        plt.title("L'immagine ritrae dei ciottoli")
    
if __name__=='__main__':
    main()
    plt.show()