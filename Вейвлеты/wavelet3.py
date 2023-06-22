import keyboard
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#from scipy import fft


def haara_wavelet(ys):
    coef = 1/np.sqrt(2)
    N = ys.shape[0]
    k = int(np.log2(N)) + 1
    Cs = [ys.copy()]
    Ds = []
    for i in range(1,k):
        N = N//2
        cs = np.zeros(N)
        ds = np.zeros(N)
        for j in range(N):
            cs[j] = coef*(Cs[i-1][2*j] + Cs[i-1][2*j+1])
            ds[j] = coef*(Cs[i-1][2*j] - Cs[i-1][2*j+1])
        Cs.append(cs) 
        Ds.append(ds)
    return Cs, Ds

def cs_normalize(Cs, heightmul=100, weightmul=2):
    n = 2
    #N = Cs[0].shape[0]
    k = len(Cs)
    for i in range(1,k):
        Cs[i] = np.repeat(Cs[i], n)
        n *= 2
    return np.repeat(
        np.repeat(np.array(Cs), heightmul, axis=0),
        weightmul, axis=1)
    

def get_noramlize_wavelet(ys):
    return create_waveletmesh(ys,1,1)

def create_waveletmesh(ys, hm=100, wm=2):
    CS_norm = cs_normalize(haara_wavelet(ys)[0], hm, wm)
    DS_norm = cs_normalize(haara_wavelet(ys)[1], hm, wm)
    return CS_norm, DS_norm

def wavelet_plot(ax, cs, is_ds=False):
    if is_ds:
        cs = np.vstack([np.zeros(cs.shape[1]), cs])
    ax.imshow(cs, aspect="auto", interpolation="nearest")
    ax.invert_yaxis()
    if is_ds:
        ax.set_ylim(ymin=0.5)

def FM(xs, ys, w, m=1):
    integral = 0
    integrated_xs = np.zeros(ys.shape[0])
    for i in range(1, len(ys)):
        area = (ys[i]+ys[i-1])*(xs[i]-xs[i-1])/2
        integral = (integral + m*area) % (2*np.pi)
       
        integrated_xs[i] = integral
    return np.cos(w*xs+integrated_xs)
        

def PM(xs, ys, w, m=4):
    return np.cos(w*xs+m*ys)


def AM(xs, ys, w, m=0.5):
    return np.cos(w*xs)*(1+m*ys)


xs = np.linspace(0,10.23,1024)
ys = np.sin(xs)



fig, axs = plt.subplots(2,3)
axs[0,0].set_title("АМ")
axs[0,1].set_title("C")
axs[0,2].set_title("D")
cs, ds = get_noramlize_wavelet(ys)
axs[0,0].plot(xs, ys)
wavelet_plot(axs[0,1], cs)
wavelet_plot(axs[0,2], ds, True)
am =  AM(xs,ys, 10)
cs, ds = get_noramlize_wavelet(am)
axs[1,0].plot(xs, am)
wavelet_plot(axs[1,1], cs)
wavelet_plot(axs[1,2], ds, True)
plt.show()


fig, axs = plt.subplots(2,3)
axs[0,0].set_title("ЧМ")
axs[0,1].set_title("C")
axs[0,2].set_title("D")
cs, ds = get_noramlize_wavelet(ys)
axs[0,0].plot(xs, ys)
wavelet_plot(axs[0,1], cs)
wavelet_plot(axs[0,2], ds, True)
fm =  FM(xs,ys, 10, 4)
cs, ds = get_noramlize_wavelet(fm)
axs[1,0].plot(xs, fm)
wavelet_plot(axs[1,1], cs)
wavelet_plot(axs[1,2], ds, True)
plt.show()


fig, axs = plt.subplots(2,3)
axs[0,0].set_title("ФМ")
axs[0,1].set_title("C")
axs[0,2].set_title("D")
cs, ds = get_noramlize_wavelet(ys)
axs[0,0].plot(xs, ys)
wavelet_plot(axs[0,1], cs)
wavelet_plot(axs[0,2], ds, True)
pm =  PM(xs,ys, 10, 4)
cs, ds = get_noramlize_wavelet(pm)
axs[1,0].plot(xs, pm)
wavelet_plot(axs[1,1], cs)
wavelet_plot(axs[1,2], ds, True)
plt.show()


fig, axs = plt.subplots(2,3)
axs[0,0].set_title("ФМн")
axs[0,1].set_title("C")
axs[0,2].set_title("D")
ys = np.repeat([1,0,1,0,1,0,1,0],128)
cs, ds = get_noramlize_wavelet(ys)
axs[0,0].plot(xs, ys)
wavelet_plot(axs[0,1], cs)
wavelet_plot(axs[0,2], ds, True)
pm =  PM(xs,ys, (7*np.pi/2)/(128*(xs[1]-xs[0])), np.pi)
cs, ds = get_noramlize_wavelet(pm)
axs[1,0].plot(xs, pm)
wavelet_plot(axs[1,1], cs)
wavelet_plot(axs[1,2], ds, True)
plt.show()
