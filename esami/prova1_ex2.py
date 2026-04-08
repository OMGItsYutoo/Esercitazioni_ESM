import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
from skimage.color import rgb2hsv, hsv2rgb

def main():
    plt.close('all')

    x=np.float64(io.imread("images/pears_noise.png"))/255
    
    x_hsv=rgb2hsv(x)
    
    H=x_hsv[:,:,0]
    S=x_hsv[:,:,1]
    V=x_hsv[:,:,2]
    
    v_fft=np.fft.fftshift(np.fft.fft2(V))
    
    m=np.fft.fftshift(np.fft.fftfreq(V.shape[0]))
    n=np.fft.fftshift(np.fft.fftfreq(V.shape[1]))
    l,k=np.meshgrid(n,m)
    
    mu_1=0.1;nu_1=-0.1;B=0.025
    mu_2=0.2;nu_2=-0.2;B=0.025
    
    D1=np.sqrt((k-mu_1)**2+(l-nu_1)**2)
    D2=np.sqrt((k+mu_1)**2+(l+nu_1)**2)
    D3=np.sqrt((k-mu_2)**2+(l-nu_2)**2)
    D4=np.sqrt((k+mu_2)**2+(l+nu_2)**2)
    F=(D1>B) & (D2>B) & (D3>B) & (D4>B)
    Y=v_fft*F

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(D1,cmap='gray',clim=[0,1])
    plt.subplot(1,3,2)
    plt.imshow(D2,cmap='gray',clim=[0,1])
    plt.subplot(1,3,3)
    plt.imshow(F,cmap='gray',clim=[0,1])

    plt.figure()
    plt.imshow(np.log(1+np.abs(Y)), cmap='gray', extent=(-0.5,+0.5,+0.5,-0.5))
    
    V_filtered=np.real(np.fft.ifft2(np.fft.fftshift(Y)))

    
    y=hsv2rgb(np.stack((H,S,V_filtered),2))
    
    
    S=S**0.6
    y_enh=hsv2rgb(np.stack((H,S,V_filtered),2))

    R=y[:,:,0]
    G=y[:,:,1]
    B=y[:,:,2]
    
    gamma=1.2
    R=R**gamma
    G=G**gamma
    B=B**gamma
    
    y_enh2=np.stack((R,G,B),2)
    
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(x)
    plt.subplot(1,4,2)
    plt.imshow(y)
    plt.subplot(1,4,3)
    plt.imshow(y_enh)
    plt.subplot(1,4,4)
    plt.imshow(y_enh2)
    
if __name__=='__main__':
    main()
    plt.show()