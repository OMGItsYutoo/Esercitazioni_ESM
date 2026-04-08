import skimage.io as io
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from skimage.exposure import equalize_hist,histogram

def main():
    plt.close('all')
    x=np.float64(io.imread('images/leno.jpg'))
    # if x.ndim == 3:
    #     x = 0.299 * x[:,:,0] + 0.587 * x[:,:,1] + 0.114 * x[:,:,2]
    # else:
    #     x= x
    X=np.fft.fft2(x)
    X=np.fft.fftshift(X)
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(np.log(1+np.abs(X)),clim=None,cmap='gray',extent=(-0.5,+0.5,+0.5,-0.5))
    plt.subplot(1,3,3)
    plt.imshow(np.angle(X),clim=None,cmap='gray',extent=(-0.5,+0.5,+0.5,-0.5))
    
    y_mod=np.fft.ifft2(np.fft.ifftshift(np.abs(X))).real
    y_ph=np.fft.ifft2(np.exp(1j*np.fft.ifftshift(np.angle(X)))).real
    
    # y_mod=equalize_hist(y_mod)
    # y_ph=equalize_hist(y_ph)
    
    plt.figure(2)
    plt.subplot(1,3,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(np.log(1+y_mod),clim=None,cmap='gray')
    plt.subplot(1,3,3)   
    plt.imshow(y_ph,clim=None,cmap='gray')
    
    # Y = np.log(1+np.abs((X)))
    # m = np.fft.fftshift(np.fft.fftfreq(Y.shape[0]))
    # n = np.fft.fftshift(np.fft.fftfreq(Y.shape[1]))
    
    # ax = plt.figure().add_subplot(projection='3d')
    # l,k=np.meshgrid(n,m)
    # ax.plot_surface(l,k,Y,linewidth=0,cmap='jet')
    
    plt.show()
    
if __name__=='__main__':
    main()
    #script 17