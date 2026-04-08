import numpy as np
import scipy.ndimage as ndi
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb

def fast_var(x,k):
    return ndi.uniform_filter(x**2,(k,k))-ndi.uniform_filter(x,(k,k))**2

def main():
    plt.close('all')
    
    x1=np.float64(io.imread('images/disk1.gif'))/255
    x2=np.float64(io.imread('images/disk2.gif'))/255
    
    #Gif Squeezing
    x1=np.squeeze(x1) # Trasforma (1, 480, 640, 3) in (480, 640, 3)
    x2=np.squeeze(x2)
    
    x1_hsv=rgb2hsv(x1)
    x2_hsv=rgb2hsv(x2)
    
    H=x1_hsv[:,:,0]
    S=x1_hsv[:,:,1]
    V_1=x1_hsv[:,:,2]
    V_2=x2_hsv[:,:,2]
    
   
    L=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    lap1_sqr=ndi.correlate(V_1,L)**2
    lap2_sqr=ndi.correlate(V_2,L)**2
    
    k=5
    act_lvl1=ndi.uniform_filter(lap1_sqr,(k,k))*fast_var(lap1_sqr,k)
    act_lvl2=ndi.uniform_filter(lap2_sqr,(k,k))*fast_var(lap2_sqr,k)
    
    act_lvl_sum=act_lvl1+act_lvl2
    act_lvl1_norm=act_lvl1/act_lvl_sum
    act_lvl2_norm=act_lvl2/act_lvl_sum
    V_y=act_lvl1_norm*V_1+act_lvl2_norm*V_2
    
    y_hsv=np.stack((H,S,V_y),2)
    y_rgb = hsv2rgb(y_hsv)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x1)
    plt.subplot(1,3,2)
    plt.imshow(x2)
    plt.subplot(1,3,3)
    plt.imshow(y_rgb)
    
if __name__=='__main__':
    main()
    plt.show()



