import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

def main():
    plt.close('all')
    
    x=np.float64(io.imread('images/car.tif'))
    X=np.fft.fftshift(np.fft.fft2(x))
    
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(np.log(1+np.abs(X)),clim=None,cmap='gray',extent=(-0.5,+0.5,+0.5,-0.5))
    plt.subplot(1,3,3)
    plt.imshow(np.angle(X),clim=None,cmap='gray',extent=(-0.5,+0.5,+0.5,-0.5))
    
    m=np.fft.fftshift(np.fft.fftfreq(x.shape[0]))
    n=np.fft.fftshift(np.fft.fftfreq(x.shape[1]))
    l,k=np.meshgrid(n,m)
    
    
    
    nu = 0.1635 
    mu1 = 0.165  
    mu2 = 0.33 
    B = 0.04
    
    D1 = np.sqrt((l-nu)**2 + (k-mu1)**2)  
    D2 = np.sqrt((l+nu)**2 + (k+mu1)**2)  
    D3 = np.sqrt((l-nu)**2 + (k+mu1)**2)  
    D4 = np.sqrt((l+nu)**2 + (k-mu1)**2)  
    
    
    D5 = np.sqrt((l-nu)**2 + (k-mu2)**2)
    D6 = np.sqrt((l+nu)**2 + (k+mu2)**2)
    D7 = np.sqrt((l-nu)**2 + (k+mu2)**2)
    D8 = np.sqrt((l+nu)**2 + (k-mu2)**2)
    
    H = (D1>B) & (D2>B) & (D3>B) & (D4>B) & (D5>B) & (D6>B) & (D7>B) & (D8>B)
    
    ax = plt.figure().add_subplot(projection='3d')
    l,k=np.meshgrid(n,m)
    ax.plot_surface(l,k,H,linewidth=0,cmap='jet')
    
    # plt.figure()
    # plt.imshow(np.log(1+np.abs(H)),clim=None,cmap='gray',extent=(-0.5,+0.5,+0.5,-0.5))
    
    Y=X*H
    y=np.real(np.fft.ifft2(np.fft.fftshift(Y)))
    plt.figure()
    plt.imshow(np.log(np.abs(Y)+1),clim=None,cmap='gray')
    plt.figure()
    plt.imshow(y,clim=[0,255],cmap='gray')
    
    ax = plt.figure().add_subplot(projection='3d')
    l,k=np.meshgrid(n,m)
    ax.plot_surface(l,k,np.log(1+np.abs(Y)),linewidth=0,cmap='jet')
    
    plt.show()
    
if __name__=='__main__':
    main()