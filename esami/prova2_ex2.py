import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import skimage.io as io

def main():
    plt.close('all')
    
    x=np.float64(io.imread("images/fiori.jpg"))
    
    m=np.fft.fftshift(np.fft.fftfreq(x.shape[0]))
    n=np.fft.fftshift(np.fft.fftfreq(x.shape[1]))
    l,k=np.meshgrid(n,m)
    
    mu=0.25;nu=0;B=0.15
    D=np.abs(np.sqrt(l**2+k**2)-mu)
    H=D<(B/2)
    
    #Passa basso decentrato
    # D1=np.sqrt((k-mu)**2+(l-nu)**2)
    # D2=np.sqrt((k+mu)**2+(l+nu)**2)
    # M,N=x.shape[0:2]
    # H=np.ones(shape=(M,N),dtype=np.float64)
    # H[(D1>B) & (D2>B)]=0.0
    
    plt.figure()
    plt.imshow(H, clim=[0,1], cmap='gray',extent=(-0.5, 0.5, -0.5, 0.5))
    
    ax=plt.figure().add_subplot(projection='3d')
    l,k=np.meshgrid(n,m)
    ax.plot_surface(l,k,H,linewidth=0,cmap='jet')
    
    R=x[:,:,0]
    G=x[:,:,1]
    Bl=x[:,:,2]
    
    R_f=np.fft.fftshift(np.fft.fft2(R))
    G_f=np.fft.fftshift(np.fft.fft2(G))
    B_f=np.fft.fftshift(np.fft.fft2(Bl))
    
    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(np.log(1+np.abs(R_f)),clim=None,cmap='gray')
    # plt.subplot(1,3,2)
    # plt.imshow(np.log(1+np.abs(G_f)),clim=None,cmap='gray')
    # plt.subplot(1,3,3)
    # plt.imshow(np.log(1+np.abs(B_f)),clim=None,cmap='gray')
    
    Rn=np.fft.ifft2(np.fft.ifftshift(R_f*H))
    Gn=np.fft.ifft2(np.fft.ifftshift(G_f*H))
    Bn=np.fft.ifft2(np.fft.ifftshift(B_f*H))
    
    y=np.real(np.stack((Rn,Gn,Bn), 2))
    
    plt.figure()
    plt.imshow(y)
    
    Bs=[0.05,0.10,0.15,0.20]
    
    psnr_list=[]
    
    for b in Bs:
        D=np.abs(np.sqrt(l**2+k**2)-mu)
        H=D<(b/2)
        
        Rn=np.fft.ifft2(np.fft.ifftshift(R_f*H))
        Gn=np.fft.ifft2(np.fft.ifftshift(G_f*H))
        Bn=np.fft.ifft2(np.fft.ifftshift(B_f*H))    
        
        y=np.real(np.stack((Rn,Gn,Bn), 2))

        mse=np.mean((x-y)**2)
        psnr=10*np.log10(255**2/mse)
        #snr=10*np.log10(np.var(x)/mse)
        
        psnr_list.append(psnr)
    
    plt.figure()
    plt.plot(Bs,psnr_list,'r-o')
    plt.xlabel("B values")
    plt.ylabel("PSNR")
    plt.grid('on')
    
if __name__=='__main__':
    main()
    plt.show()