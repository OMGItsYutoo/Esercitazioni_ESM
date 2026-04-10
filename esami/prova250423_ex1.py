import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import rgb2hsv, hsv2rgb

def color_balancing(x):
    x_cmy=1-x
    
    C=x_cmy[:,:,0]
    M=x_cmy[:,:,1]
    Y=x_cmy[:,:,2]
    
    C_elab=C**1.5
    M_elab=M**1
    Y_elab=Y**1

    return 1-np.stack((C_elab,M_elab,Y_elab),2)

def filtra(x):
    
    m=np.fft.fftshift(np.fft.fftfreq(x.shape[0]))
    n=np.fft.fftshift(np.fft.fftfreq(x.shape[1]))
    l,k=np.meshgrid(n,m)
    
    mu=0.1;nu=0.25
    H=(np.abs(k)<mu) & (np.abs(l)<=nu)
    
    plt.figure()
    plt.imshow(H,cmap='gray',clim=[0,1])
    
    ax=plt.figure().add_subplot(projection='3d')
    m=np.fft.fftshift(np.fft.fftfreq(x.shape[0]))
    n=np.fft.fftshift(np.fft.fftfreq(x.shape[1]))
    l,k=np.meshgrid(n,m)
    ax.plot_surface(l,k,H,linewidth=0,cmap='jet')
    
    x_hsv=rgb2hsv(x)
    h=x_hsv[:,:,0]
    s=x_hsv[:,:,1]
    v=x_hsv[:,:,2]
    
    V=np.fft.fftshift(np.fft.fft2(v))
    
    Vn=V*H
    
    vn=np.real(np.fft.ifft2(np.fft.ifftshift(Vn)))
    
    return hsv2rgb(np.stack((h,s,vn),2))
    
def main():
    plt.close('all')
    
    x=np.float64(io.imread('images/foto.jpg'))/255
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x)
    
    y=color_balancing(x)
    
    plt.subplot(1,3,2)
    plt.imshow(y)
    
    y_f=filtra(y)
    
    plt.subplot(1,3,3)
    plt.imshow(y_f)
    
if __name__=='__main__':
    main()
    plt.show()