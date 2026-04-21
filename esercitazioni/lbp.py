import skimage.data as data
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from skimage.feature import local_binary_pattern

def main():

    x=np.float64(data.brick())
    plt.figure()
    plt.imshow(x, clim=[0,255], cmap='gray')
    
    h0 = np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=np.float64)
    h1 = np.array([[0,1,0],[0,-1,0],[0,0,0]], dtype=np.float64)
    h2 = np.array([[0,0,1],[0,-1,0],[0,0,0]], dtype=np.float64)
    h3 = np.array([[0,0,0],[0,-1,1],[0,0,0]], dtype=np.float64)
    h4 = np.array([[0,0,0],[0,-1,0],[0,0,1]], dtype=np.float64)
    h5 = np.array([[0,0,0],[0,-1,0],[0,1,0]], dtype=np.float64)
    h6 = np.array([[0,0,0],[0,-1,0],[1,0,0]], dtype=np.float64)
    h7 = np.array([[0,0,0],[1,-1,0],[0,0,0]], dtype=np.float64)
    b0 = ndi.correlate(x, h0) >= 0
    b1 = ndi.correlate(x, h1) >= 0
    b2 = ndi.correlate(x, h2) >= 0
    b3 = ndi.correlate(x, h3) >= 0
    b4 = ndi.correlate(x, h4) >= 0
    b5 = ndi.correlate(x, h5) >= 0
    b6 = ndi.correlate(x, h6) >= 0
    b7 = ndi.correlate(x, h7) >= 0
    
    y = b0 + b1*2 + b2*4 + b3*8 + b4*16 + b5*32 + b6*64 + b7*128
    
    plt.figure(); plt.imshow(y, clim=[0,255], cmap="gray")
    plt.title("immagine LBP")
    
    #Possiamo identificare le patch uniformi dell’immagine sapendo che la descrizione relativa `e 0 o 255.
    mappa_uniformi = (y==0) | (y==255)
    dlb=[15,30,60,120,240,255,195,135]
    mappa_bordi = np.isin(y, dlb)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(mappa_uniformi, clim=[0,1], cmap="gray")
    plt.title("mappa delle patch uniformi")
    plt.subplot(1,2,2)    
    plt.imshow(mappa_bordi, clim=[0,1], cmap="gray")
    plt.title("mappa dei bordi")
    
    hist,_=np.histogram(y.flatten(),np.arange(257), density=True)  #density=true restituisce l'istogramma normalizzato
    plt.figure()
    plt.bar(np.arange(256),hist)

    #LBP circolare
    P=8;R=1
    yf=local_binary_pattern(x,P,R,method='ror')
    hist,_=np.histogram(yf.flatten(),np.arange(257), density=True)  #density=true restituisce l'istogramma normalizzato
    plt.figure()
    plt.subplot(1,2,2)
    plt.bar(np.arange(256),hist)
    plt.subplot(1,2,1)
    plt.imshow(yf,clim=[0,255], cmap="gray")
    
if __name__=='__main__':
    main()
    plt.show()