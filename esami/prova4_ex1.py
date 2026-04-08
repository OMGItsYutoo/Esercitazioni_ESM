import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
from bitop import bitset, bitget

def main():
    plt.close('all')
    
    x=np.reshape(np.fromfile("images/upupa.y", dtype=np.uint8),(256,512))
    sign=np.reshape(np.fromfile("images/firma.y", dtype=np.uint8),(256,512))
    
    sign_bitplane=1
    x=bitset(x,sign_bitplane,sign)

    plt.figure()
    plt.imshow(x,clim=[0,255], cmap='gray')
    
    plt.figure()
    plt.imshow(sign,clim=[0,1], cmap='gray')
    
    quality_list=[10,20,30,40,50,60,70,80,90,100]
    mse_list=[]
    
    for q in quality_list:
        io.imsave('images/upupa.jpg', x, quality=q)
        x_jpg=io.imread("images/upupa.jpg")
        sign_jpg=bitget(x_jpg,sign_bitplane)
        mse=np.mean((sign-sign_jpg)**2)
        mse_list.append(mse)
            
    plt.figure()
    plt.plot(quality_list,mse_list,'r-o')
    plt.xlabel('Quality level')
    plt.ylabel('MSE(Q)')
    plt.grid('on')

    x=np.float64(x)
    X=np.fft.fftshift(np.fft.fft2(x))
    
    m=np.fft.fftshift(np.fft.fftfreq(x.shape[0]))
    n=np.fft.fftshift(np.fft.fftfreq(x.shape[1]))
    l,k=np.meshgrid(n,m)
    
    B=[0.01,0.1,0.2,0.3,0.4]
    mse_list_freq=[]
    D=np.sqrt(k**2+l**2)
    
    for b in B:    
        H=D<B[0]
        Y=H*X
        y=np.real(np.fft.ifft2(np.fft.ifftshift(Y)))
        sign_freq=bitget(np.uint8(y),sign_bitplane)
        mse=np.mean((sign-sign_freq)**2)
        mse_list_freq.append(mse)

    plt.figure()
    plt.plot(B,mse_list_freq,'r-o')
    plt.xlabel(r'Cut-off Frequency ($D_0$)')
    plt.ylabel(r'MSE')
    plt.grid('on')

if __name__=='__main__':
    main()
    plt.show()





