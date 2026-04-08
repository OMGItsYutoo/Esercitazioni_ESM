import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io

def respfreq(h):
    #allineo la matrice al centro della figura N*N su cui effettuo poi la trasformata
    #non effettuo la trasformata sul blocchetto 5*5 perché altrimenti dovrei effettuare una interpolazione cardinale in fase di visualizzazione
    N=512
    h_full=np.zeros((N, N)) 
    kh,kw=h.shape
    
    h_full[:kh,:kw]=h
    
    h_full=np.roll(h_full, -(kh//2), axis=0)
    h_full=np.roll(h_full, -(kw//2), axis=1)

    #nel caso mi interessasse solo della visualizzazione della fft del filtro potrei anche fare  H = np.fft.fft2(h_full,(N,N))
    H=np.fft.fft2(h_full)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(h,cmap='gray',clim=[0,1])
    plt.subplot(1,2,2)
    plt.imshow(np.log(1+np.abs(H)),cmap='gray',extent=(-0.5,+0.5,-0.5,+0.5))

    ax=plt.figure().add_subplot(projection='3d')
    m=np.fft.fftshift(np.fft.fftfreq(512))
    n=np.fft.fftshift(np.fft.fftfreq(512))
    l,k=np.meshgrid(n,m)
    ax.plot_surface(l,k,np.log(1+np.abs(H)),linewidth=0,cmap='jet')
    
    return np.fft.ifftshift(H)

def filtra(x,h):
    
    y=ndi.correlate(x,h,mode='wrap') #mode=wrap per fare il match con la natura periodica della fft
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(y,clim=[0,255],cmap='gray')
    
    return y

def filtrafreq(x,H):
    
    X=np.fft.fft2(x)
    
    Y=X*H
    
    y=np.real(np.fft.ifft2(Y))
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(x,clim=[0,255],cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(y,clim=[0,255],cmap='gray')
    
    return y 

def main():
    plt.close('all')
    
    x=np.float64(io.imread("images/barbara.png"))
    
    print(x.shape)
    h=np.zeros((5, 5))
    h[0, 0] = 1
    h[0, 4] = 1
    h[4, 0] = 1
    h[4, 4] = 1
    h[2, 2] = -4

    H=respfreq(h)
    
    spaz=filtra(x,h)
    
    freq=filtrafreq(x,H)
    
    mse=np.mean((spaz-freq)**2)
    
    print(f"MSE: {mse}")
    
if __name__=='__main__':
    main()
    plt.show()