import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import keyboard
import os
from scipy import fft

print("Press e")
keyboard.wait("e")
os.system('cls')

#Вейвлет Хаара
def haara_wavelet(ys):
    coef = 1/np.sqrt(2)# это 1/корень из 2. нужен для вычисления параметров С
    N = ys.shape[0]
    k = int(np.log2(N)) + 1# ХЗ зачем это, но k используется в цикле
    Cs = [ys.copy()]#в Cs зачем-то копируется функция???????????
    Ds = []#пустой массив?????????
#начинается цикл заполнения массива коэффициентов, пока непонятно, зачем использовать k, но вероятно связано с созданием пар
    for i in range(1,k):
        N = N//2#делим надвое, тк по алгоритму будем работать с парами(1,2; 3,4;и тд)
        cs = np.zeros(N)#это заполнение нулями, хз зачем
        ds = np.zeros(N)
        for j in range(N):
            cs[j] = coef*(Cs[i-1][2*j] + Cs[i-1][2*j+1])#тут всё понятно, всё по формуле: (х0+х1)/корень из 2
            ds[j] = coef*(Cs[i-1][2*j] - Cs[i-1][2*j+1])#всё по формуле: (х0-х1)/корень из 2
        Cs.append(cs)#присоединение созданных коэффицентов к существующему массиву. НО- массив коэффициентов тут будет как 1 элемент. типа: [1,2,3,(1,2,3,4,5,6)]- 4 элемента в массиве
        Ds.append(ds)
    return Cs, Ds


#Нормировка
#нужна для того, чтобы сократить разбег значений?????????????
#возможно эта часть- про цветную карту, поэтому тут играют роль высота и ширина, но ХЗ
def cs_normalize(Cs, heightmul=100, weightmul=2):
    n = 2
    #N = Cs[0].shape[0]
    k = len(Cs)#Вычисление размера массива
    for i in range(1,k):
        Cs[i] = np.repeat(Cs[i], n)#Это повторение элемент Cs[i] повторяется n раз
        n *= 2
    return np.repeat(
        np.repeat(np.array(Cs), heightmul, axis=0),#axis=0- операция идёт по строкам
        weightmul, axis=1)#axis=1- операция идёт по столбцам

#Создание Вейвлета с помощью двух предыдущих функций
#Принимается функция, именуется ys и отправляется в функцию cs_normalize, которая в свою очередь использует ф-ю haara_wavelet
def create_waveletmesh(ys, hm=100, wm=2):
    return cs_normalize(haara_wavelet(ys)[0], hm, wm), cs_normalize(haara_wavelet(ys)[1], hm, wm)

#х0 и х1- разбег значений (откуда и куда). N- количество значений
x0=0
x1=7
N=1024

xs=np.linspace(x0,x1,N)#тут типа массив, который заполняется N- количеством значений от х0 до х1
ns=np.arange(0,N/2)

#Вычисление функции f
fs=np.sin(2*np.pi*xs)+np.sin(2*np.pi*xs)#Вычисляется функция f, где 2*np.pi*xs- фи 
print("Функция f:", sep="\n")
print (*fs, sep="\n")
print("Press e")
keyboard.wait("e")
os.system('cls')

#Вычисление Фурье для f
s_fs=np.abs(fft.fft(fs))[:(N//2)]#вычисляется Ф преобр-е, где fft-команда из библиотеки, [:(N//2)]- срез(тоесть N сокращается вдвое)
print("Фурье преобразование для f:", sep="\n")
print (*s_fs, sep="\n")
print("Press e")
keyboard.wait("e")
os.system('cls')

#Вычисление функции g
#3,0- у нас в формуле параметр а. От него зависит когда будет скачок в графике
gs=np.sin(2*np.pi*xs)+np.heaviside(xs-3,0)*np.sin(2*np.pi*xs)#та же формула, что и f, но исользуется функция Хевисайда в середине 
print("Функция g:", sep="\n")
print (*gs, sep="\n")
print("Press e")
keyboard.wait("e")
os.system('cls')

#Вычисление Фурье для g
s_gs=np.abs(fft.fft(gs))[:(N//2)]
print("Фурье преобразование для g:", sep="\n")
print (*s_gs, sep="\n")
print("Press e")
keyboard.wait("e")
os.system('cls')

#Вывод граффиков функций и их Ф.преобразований
fig, axs=plt.subplots(2,2)#создание фигуры, где будут отображаться графики
axs[0][0].plot(xs,fs)#слева вверху- график f
axs[0][1].plot(xs,gs)#справа вверху- график g
axs[1][0].plot(ns,s_fs)#слева внизу- Ф f
axs[1][1].plot(ns,s_gs)#справа внизу- Ф g
plt.show()

#Запускаем создание Вейвлета и рисуем карты
fig, axs = plt.subplots(2,2)#создание фигуры 2х2
axs[0][0].plot(xs, fs)#слева вверху- график f
axs[1][0].plot(xs, gs)#слева внизу- график g

cfs = create_waveletmesh(fs)[0][::-1]#вызов вейвлет-функции для f
cgs = create_waveletmesh(gs)[0][::-1]
dfs = create_waveletmesh(fs)[1][::-1]
dgs = create_waveletmesh(gs)[1][::-1]

vmin = min(cfs.min(), cgs.min())
vmax = max(cfs.max(), cgs.max())

im = axs[0][1].imshow(cfs, vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=axs[0,1]);
axs[0][1].set_xticks([1,512,1024,1536,2048])
axs[0][1].set_xticklabels(["1","256","512","768","1024"])
axs[0][1].set_yticks(
     [50,150,250,350,450,550,650,750,850,950,1050])
axs[0][1].set_yticklabels(["0","1","2","3","4","5","6","7","8","9","10"][::-1])

im = axs[1][1].imshow(cgs, vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=axs[1,1]);
axs[1][1].set_xticks([1,512,1024,1536,2048])
axs[1][1].set_xticklabels(["1","256","512","768","1024"])
axs[1][1].set_yticks(
    [50,150,250,350,450,550,650,750,850,950,1050])
axs[1][1].set_yticklabels(["0","1","2","3","4","5","6","7","8","9","10"][::-1])

plt.show()

#Восстановление сигнала в картинке
def restore_signal(Cn, Ds):
    coef = 1/np.sqrt(2)
    Ds = Ds[::-1]
    cs_prev = [Cn]
    cs = []
    for ds in Ds:
        for i in range(len(ds)):

            cs.extend([coef*(cs_prev[i]+ds[i]),
                       coef*(cs_prev[i]-ds[i])])
        cs_prev = cs
        cs = []
    return cs_prev



fig, axs = plt.subplots(2,2)

axs[0][0].plot(xs, fs)
axs[1][0].plot(xs, gs)

vmin = min(cfs.min(), cgs.min())
vmax = max(cfs.max(), cgs.max())
im = axs[0][1].imshow(dfs, vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=axs[0,1]);
axs[0][1].set_xticks([1,512,1024])
axs[0][1].set_xticklabels(["1","256","512"])
axs[0][1].set_yticks(
     [50,150,250,350,450,550,650,750,850,950])
axs[0][1].set_yticklabels(["1","2","3","4","5","6","7","8","9","10"][::-1])

im = axs[1][1].imshow(dgs, vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=axs[1,1]);
axs[1][1].set_xticks([1,512,1024])
axs[1][1].set_xticklabels(["1","256","512"])
axs[1][1].set_yticks(
     [50,150,250,350,450,550,650,750,850,950])
axs[1][1].set_yticklabels(["1","2","3","4","5","6","7","8","9","10"][::-1])

plt.show()

#Восстановление сигнала в графике
raw_cfs, raw_dfs = haara_wavelet(fs); #Берем cn и ds
cn = raw_cfs[-1][0] #Берем последний Cn
restore_fs = restore_signal(cn, raw_dfs)
raw_cgs, raw_dgs = haara_wavelet(gs); #Берем cn и ds
cn = raw_cgs[-1][0] #Берем последний ds
restore_gs = restore_signal(cn, raw_dgs)
fig, axs = plt.subplots(2)
axs[0].plot(xs, restore_fs)
axs[0].plot(xs, fs, color="green")
axs[1].plot(xs, restore_gs)
axs[1].plot(xs, gs, color="green")

plt.show()

print("Всё прошло хорошо. Чтобы закрыть нажмите е")
keyboard.wait("e")