import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.exposure import histogram
    
def block_fun(x):
    x=np.reshape(x,(9,9))
    X=np.fft.fft2(x)
    a=np.array([np.real(X[-1,0]),np.imag(X[-1,0]),np.real(X[-1,1]),np.imag(X[-1,1]),np.real(X[0,1]),np.imag(X[0,1]),np.real(X[1,1]),np.imag(X[1,1])])
    c=a>0
    s=0
    for i in range(8):
        s+=c[i]*(2**i)
    return s
    
def main():
    plt.close('all')
    
    x1=np.float64(io.imread('images/impronta1.png'))
    x2=np.float64(io.imread('images/impronta2.png'))
    
    k=9
    y1=ndi.generic_filter(x1,block_fun,(k,k))
    y2=ndi.generic_filter(x2,block_fun,(k,k))

    hist1,_=np.histogram(y1, bins=256, range=(0, 255))
    hist2,_=np.histogram(y2, bins=256, range=(0, 255))
    
    std1=np.std(hist1)
    std2=np.std(hist2)
    impronte=[]
    impronte.append(std1)   
    impronte.append(std2)
       
    plt.figure();plt.bar(range(256),hist1);plt.axis((0,255,0,1.1*np.max(hist1)))
    plt.figure();plt.bar(range(256),hist2);plt.axis((0,255,0,1.1*np.max(hist2)))
    
    for i in range(2):
        if impronte[i]<250:
            print(f"Impronta {i+1} è falsa")
        else:
            print(f"Impronta {i+1} è vera")
            
if __name__=='__main__':
    main()
    plt.show()