
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.io as io
plt.close('all')

def blk_fun_point4(x):
    values = x[::8]       # prende un valore ogni 8
    y = np.median(values) # calcola il mediano sui valori estratti
    return y

x = np.float64(io.imread("images/aerei.png"))

plt.figure()
plt.imshow(x,clim=[0,255],cmap="gray")
plt.title("Immagine")

# H1
h_d2ver = np.array([[0,-1,0],[0,2,0],[0,-1,0]])
y = np.abs(ndi.correlate(x, h_d2ver))
y[y>50] = 0

# H2
e_s = 33*ndi.uniform_filter(y, (1,33))

# H3
e = e_s - ndi.median_filter(e_s,(33,1))

# H4
g_H = ndi.generic_filter(e, blk_fun_point4, (33,1))    

# show g_H
plt.figure()
plt.imshow(g_H, clim=None,cmap="gray")
plt.title("g_H")


# V1
h_d2hor = np.array([[0,0,0],[-1,2,-1],[0,0,0]])
y = np.abs(ndi.correlate(x, h_d2hor))
y[y>50] = 0

# V2
e_s = 33*ndi.uniform_filter(y, (33,1))

# V3
e = e_s - ndi.median_filter(e_s,(1,33))

# V4
g_V = ndi.generic_filter(e, blk_fun_point4, (1,33))    

# show g_V
plt.figure()
plt.imshow(g_V, clim=None,cmap="gray")
plt.title("g_V")

# show g
g = g_H + g_V
plt.figure()
plt.imshow(g, clim=None,cmap="gray")
plt.title("g")

plt.show()

# la griglia dovuta dalla compressione jpeg non è presente in alto
# probabilmente un aereo è stato cancellato in quella zona
