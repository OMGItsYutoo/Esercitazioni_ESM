import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import rgb2gray
from skimage.exposure import histogram

def block_fun(window):
    xc = window[4]
    
    # Mappatura secondo la figura dell'esame:
    # x0=w[0], x1=w[1], x2=w[2], x3=w[5], x4=w[8], x5=w[7], x6=w[6], x7=w[3]
    indices = [0, 1, 2, 5, 8, 7, 6, 3]
    
    y = 0
    for i in range(8):
        xi = window[indices[i]]
        # Funzione u(xi - xc)
        bit = 1 if (xi - xc) >= 0 else 0
        y += (2**i) * bit
    return y

def elab(x,k):
    return ndi.generic_filter(x,block_fun,(k,k))

def main():
    plt.close('all')
    
    x1=np.float64(io.imread("images/I1.png"))/255
    x2=np.float64(io.imread("images/I2.png"))/255
    
    x1_gray=rgb2gray(x1)
    x2_gray=rgb2gray(x2)
    
    print(x1_gray.shape)
    
    H=np.array([[-1,2,-1],
                [2,-4,2],
                [-1,2,-1]])
    
    z1=ndi.correlate(x1_gray,H)
    z2=ndi.correlate(x2_gray,H)
    
    y1=elab(z1,3)
    y2=elab(z2,3)
    
    hist1,_=np.histogram(y1, bins=256, range=(0, 255))
    hist2,_=np.histogram(y2, bins=256, range=(0, 255))
    
    std1=np.std(hist1)
    std2=np.std(hist2)
    
    print(f"I1 - Std Istogramma: {std1:.2f} -> {'VERA (1)' if std1 > 495 else 'FALSA (0)'}")
    print(f"I2 - Std Istogramma: {std2:.2f} -> {'VERA (1)' if std2 > 495 else 'FALSA (0)'}")

    plt.figure()
    plt.subplot(1,2,1); plt.bar(range(256), hist1); plt.title("Istogramma I1")
    plt.subplot(1,2,2); plt.bar(range(256), hist2); plt.title("Istogramma I2")
    
    plt.figure()
    plt.subplot(1,2,1); plt.imshow(y1,clim=[0,255],cmap='gray')
    plt.subplot(1,2,2); plt.imshow(y2,clim=[0,255],cmap='gray')
    
if __name__=='__main__':
    main()
    plt.show()