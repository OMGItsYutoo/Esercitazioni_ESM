import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

def main():
    x=np.fromfile('images/zoom.y',np.float32)
    x=np.reshape(x,(128,128))
   
    m,n=x.shape 
    plt.figure()
    plt.imshow(x, clim=[0,255], cmap='gray')
    
    H=np.array([[-1,2,-1]])
    d2=ndi.correlate(x,H)
    
    ps_var = np.sum(np.abs(d2), axis=0)
    d1=np.diff(ps_var)
    
    D=np.abs(np.fft.fftshift(np.fft.fft(d1,m-2)))
    nu_h=np.fft.fftshift(np.fft.fftfreq(m-2))

    maschera_pos = nu_h > 0
    
    nu_h_pos = nu_h[maschera_pos]
    D_pos = D[maschera_pos]
    
    max_idx_h = np.argmax(D_pos)
    nu0_h = nu_h_pos[max_idx_h]
    
    R_h = 1 / nu0_h
    print(f"Frequenza del picco trovata: {nu0_h}")
    print(f"Fattore di scala stimato (R): {R_h:.3f}")       
    
    plt.figure()
    plt.plot(nu_h,D)
    
    plt.show()
    
if __name__=='__main__':
    main()


