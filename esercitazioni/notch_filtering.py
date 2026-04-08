import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np

def main():
    plt.close('all')
    x=np.float64(io.imread('images/anelli.tif'))

    X=np.fft.fft2(x)
    X=np.fft.fftshift(X)
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(np.log(1+np.abs(X)),clim=None,cmap='gray',extent=(-0.5,+0.5,+0.5,-0.5))
    plt.subplot(1,3,3)
    plt.imshow(np.angle(X),clim=None,cmap='gray',extent=(-0.5,+0.5,+0.5,-0.5))
    
    m=np.fft.fftshift(np.fft.fftfreq(x.shape[0]))
    n=np.fft.fftshift(np.fft.fftfreq(x.shape[1]))
    l,k=np.meshgrid(n,m) # l-->distanza orizzontale dal centro; k-->distanza verticale dal centro.
    
    # Parametri del filtro
    W = 0.02  
    G = 0.02  
    H = np.ones_like(X, dtype=float)
    
    notch_zona = (np.abs(l) < W) & (np.abs(k) > G)
    H[notch_zona] = 0.0
    
    ax = plt.figure().add_subplot(projection='3d')
    l,k=np.meshgrid(n,m)
    ax.plot_surface(l,k,H,linewidth=0,cmap='jet')
    
    Y=X*H    
    y=np.real(np.fft.ifft2(np.fft.fftshift(Y)))
    plt.figure(3)
    plt.imshow(np.log(np.abs(Y)+1),clim=None,cmap='gray')
    plt.figure(4)
    plt.imshow(y,clim=[0,255],cmap='gray')
    
    Y = np.log(1+np.abs((X)))
    m = np.fft.fftshift(np.fft.fftfreq(Y.shape[0]))
    n = np.fft.fftshift(np.fft.fftfreq(Y.shape[1]))
    
    ax = plt.figure().add_subplot(projection='3d')
    l,k=np.meshgrid(n,m)
    ax.plot_surface(l,k,Y,linewidth=0,cmap='jet')
    
    plt.show()
    
if __name__=='__main__':
    main()