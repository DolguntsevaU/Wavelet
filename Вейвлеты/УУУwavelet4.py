import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import keyboard
import os
from matplotlib.widgets import Slider
import cv2

#print("Press e")
#keyboard.wait("e")
#os.system('cls')

def compress_fast(img):
    im = np.zeros((int(img.shape[0]/2), int(img.shape[1]/2)))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            row_sum = img[2*i][2*j]/2+img[2*i][2*j+1]/2
            col_sum = img[2*i+1][2*j]/2 + img[2*i+1][2*j+1]/2
            im[i][j] = row_sum/2 + col_sum/2
    return im 


#print("def compress_fast is true. Press e")
#keyboard.wait("e")
#os.system('cls')


def update_image(img, ax):
    ax.imshow(img, cmap="gray", aspect="auto", interpolation="nearest")

#print("def update_image is true. Press e")
#keyboard.wait("e")
#os.system('cls')

im_orig = cv2.imread('C:\pngegg.png', cv2.IMREAD_GRAYSCALE)

fig, ax = plt.subplots()
ax.imshow(im_orig, cmap="gray")

#print("This is so gooood. Press e")
#keyboard.wait("e")
#os.system('cls')

#НАСТРОИКА ПОЛЗУНКА
def update_compress(val):
    update_image(compress_imgs[int(val)], ax)
plt.subplots_adjust(bottom=0.25)
ax_compress = plt.axes([0.35, 0.1, 0.45, 0.03])
sl_compress = Slider(ax_compress, 'Кол-во итераций', 0.0,
                   10.0, 0, valstep=1.0)
sl_compress.on_changed(update_compress)


#print("Yes yes yes!!!!!. Press e")
#keyboard.wait("e")
#os.system('cls')

compress = compress_fast

compress_imgs = [im_orig]
for i in range(10):
    compress_imgs.append(compress(compress_imgs[i]))


plt.show()
