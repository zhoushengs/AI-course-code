import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt



img_name = "Q1.jpg"
img = cv.imread(img_name,cv.IMREAD_GRAYSCALE)


kernel_v = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], np.float32)
img_v = cv.filter2D(img, -1, kernel_v)

kernel_h = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]], np.float32)
img_h = cv.filter2D(img, -1, kernel_h)

#img_skimage = io.imread('Q1.jpg')        # skimage.io imread()-----np.ndarray,  (H x W x C), [0, 255],RGB
#t_img = img[:,:,None]
#tensor_skimage = torch.from_numpy(np.transpose(img_skimage, (2, 0, 1)))
#tensor_cv = torch.from_numpy(np.transpose(img_v, (2, 0, 1)))
#tensor_pil = torch.from_numpy(np.transpose(img_pil_1, (2, 0, 1)))

def pooling(mat,ksize,method='max',pad=False):

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result


max_v = pooling(img_v, (2,2))
max_h = pooling(img_h, (2,2))

Relu_v = np.maximum(max_v, 0)
Relu_h = np.maximum(max_h, 0)

cv.imshow('img_v.png', img_v)
cv.waitKey(0)
cv.imwrite('img_v.png',img_v)
cv.imshow('img_h.png', img_h)
cv.imwrite('img_h.png',img_h)
cv.waitKey(0)

cv.imshow('max_v.png', max_v)
cv.imwrite('max_v.png',max_v)
cv.waitKey(0)
cv.imshow('max_h.png', max_h)
cv.imwrite('max_h.png',max_h)
cv.waitKey(0)

cv.imshow('Relu_v.png', Relu_v)
cv.imwrite('Relu_v.png',Relu_v)
cv.waitKey(0)
cv.imshow('Relu_h.png', Relu_h)
cv.imwrite('Relu_h.png',Relu_h)
cv.waitKey(0)