#!/usr/bin/env python3
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.stats import multivariate_normal
from scipy import random, linalg
from sklearn.model_selection import train_test_split
import torch.optim as optim
import csv


def show(img):
    """
    Displays the tensor representation of an image.
    
    Parameters:
    ----------
    img :Tensor
        An RGB-Image as stored as a Tensor with shape (B,C,H,W)

    Returns
    ----------
    figure
        A figure displaying the image

    """
    # Tensor -> Img
    toPIL = transforms.ToPILImage()
    B,C,H,W = img.size()
    img = img.view(C,H,W)
    img = toPIL(img)
    plt.imshow(img)


    
def createShadow(img):
    """
    Creates artificial shadows on an image.
    
    Generates randomly a convex polygon. 
    The radius is approximately between a tenth and half of the image.
    Pixels within this polygon are multiplied with a random value between 0.15 and 0.9
    
    Parameters
    ----------
    img: numpy image
        A colored image
    
    Returns
    ----------
    img: numpy image (np.uint8)
        An image containing an artificial shadow 
        
    """
    img = img.astype(np.float)
    h,w,_ = np.shape(img)
    mask = np.ones((h,w), np.float)
    mask = mask.astype(np.float)
    
    shadowScalar = np.random.rand()*(0.9-0.15) + 0.15
    poly = createConvexPolygon(h,w,10,2)
    mask = cv2.fillPoly(mask,poly,shadowScalar)
    
    #mask = cv2.circle(mask,(50,50),100,shadowScalar,-1)
    mask = cv2.merge((mask,mask,mask))
    img = cv2.multiply(img,mask)
    return img.astype(np.uint8)

def createSpecularity(img):
    """
    Creates artificial specularities on an image.
    
    Generates randomly a guassian bell curve modelling specular light.
    The center of the specularity has a maximum intensity between 100 and 255.
    The specularity is added to the original image.
    
    Parameters
    ----------
    img: numpy image
        A colored image
    
    Returns
    ----------
    img
        An image conatining an artificial specularity
    
    """
    img = img.astype(np.float)
    h,w,_ = np.shape(img)
    
    x, y = np.mgrid[0:w:1, 0:h:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    mask = np.zeros((h,w), np.float)

    x_mean = np.random.randint(0,w)
    y_mean = np.random.randint(0,h)

    #x_mean = min(x_mean, w-box_w-1)
    #y_mean = min(y_mean, h-box_h-1)
    
    xx = np.random.rand()*w/5
    yy = np.random.rand()*w/5
    xy = np.random.rand()*w/200
    C = np.array([[xx, xy], [xy, yy]])
    C = C.dot(np.transpose(C))
    C = C + np.eye(2)*0.00001
    #C = random.rand(2,2)*w/10
    #C = np.dot(C,C.transpose())
    rv = multivariate_normal([x_mean, y_mean], C)
    
    
    gaussian = rv.pdf(pos)
    gaussian = gaussian.transpose()
    
    factor = gaussian[y_mean,x_mean]
    
    intensity = np.random.randint(100,256)
    gaussian = (gaussian/factor) *intensity
    
    #mask[y_mean:(y_mean+box_h),x_mean:(x_mean+box_w)] = gaussian
    mask = gaussian
    mask = cv2.merge((mask,mask,mask))
    img = cv2.add(img, mask)
    _,img = cv2.threshold(img,255,255,cv2.THRESH_TRUNC)
    return img.astype(np.uint8)

def createOcclusion(img,occlusion):
    """
    Creates artificial shadows on an image.
    
    Generates randomly a convex polygon using a circle as reference.
    The radius is approximately between a seventh and third of the image.
    Pixels within this polygon are copied from a reference image.
    
    Parameters
    ----------
    img: numpy image
        A colored image.
    occlusion: numpy image
        A colored image used to generate an occlusion.
    
    Returns
    ----------
    img: numpy image (np.uint8)
        An image containing an occlusion.
        
    """
    img = img.astype(np.float)
    h,w,_ = np.shape(img)
    
    mask = np.zeros((h,w), np.float)
    mask = mask.astype(np.float)
    
    occlusion = cv2.resize(occlusion,(w,h))
    poly = createConvexPolygon(h,w,7,3)
    mask = cv2.fillPoly(mask,poly,1)
    new_image = img * (mask[:,:,None].astype(img.dtype))
    
    locs = np.where(mask > 0)
    img[locs[0], locs[1]] = occlusion[locs[0], locs[1]]
    return img.astype(np.uint8)
    
    
def createConvexPolygon(w,h,minScale,maxScale):
    """
    Randomly generates a convex polygon.
    
    Generates a random circle with a center within the image
    and a radius between 1/minScale*min(w,h) and 1/maxScale*min(w,h).
    Randomly generates 3 to 20 points on the circle. The points
    define the corners of the polygon.
    
    Parameters
    ----------
    w: int
        width of an image
    h: int
        height of an image
    minScale: float
        The minimal size of the radius is 1/minScale*min(h,w)
    maxScale: float
        The maximal size of the radius is 1/maxScale*min(h,w)
    
    Returns
    ----------
    pts: list
        pts is a list containing a numpy array. 
        The array has shape (k,2) and contains 2D points representing the polygon.
    
    """
    k = np.random.randint(3,20)
    
    x_mean = np.random.randint(0,h)
    y_mean = np.random.randint(0,w)
    
    size = min(w,h)
    r = np.random.randint(int(size/minScale),int(size/maxScale))
    
    stepSize = 2*np.pi/k
    
    angles = np.random.rand(k)*2*np.pi
    angles = np.sort(angles)
    
    x = x_mean + np.cos(angles)*r
    y = y_mean + np.sin(angles)*r
    pts = np.stack((x,y),axis=1)
    pts = np.array(pts, 'int32')
    return [pts]
        
def createMisalignment(img):
    """
    Randomly creates a shift of an image.
    The shift is between -5 and 5 pixels in the horizontal and vertical direction.
   
    Parameters
    ----------
    img: numpy image
        An input image

    
    Returns
    ----------
    img: numpy image
        A shifted image
    
    """
    w,h,_ = np.shape(img)
    offset_x = np.random.randint(-5,5)
    offset_y = np.random.randint(-5,5)
    M = np.float32([[1,0,offset_x],[0,1,offset_y]])
    img = cv2.warpAffine(img,M,(h,w))
    return img.astype(np.uint32)

def createPolygon(w, h):
    """
    [Deprecated] 
    Randomly generates 3 to 10 points representing a polygon.
    
     Parameters
    ----------
    w: int
        width of an image
    h: int
        height of an image
    """
    k = np.random.randint(3,10)
    x = np.random.randint(0,w,k)
    y = np.random.randint(0,h,k)
    pts = np.stack((x,y),axis=1)
    pts = np.array(pts, 'int32')
    return [pts]

def createDirectory(path):
    """
    Creates a directory for the given path.
    """
    if(os.path.isdir(path)):
        return
    os.mkdir(path)
    
    

def write(path,data):
    """
    Writes data to a csv file. Creates the file if it does not exist.
    Appends data to the file if file already exists.
    
    Parameters
    ----------
    path: str
        Path of the file
    data: list
        List containing row elements
    """
    with open(path, mode='a') as error_file:
        writer = csv.writer(error_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data)
        
def plot(images):
    """
    Plots multiple images.
    
    Parameters
    ----------
    images: list of tensors
        A list of images stored as tensor with shapes (1,C,H,W)
    """
    toPIL = transforms.ToPILImage()
    plt.figure(figsize=(30,30))
    columns = len(images)
    for i, img in enumerate(images):
        plt.subplot(1, columns, i + 1)
        toPIL = transforms.ToPILImage()
        B,C,H,W = img.size()
        img = img.view(C,H,W)
        img = toPIL(img)
        plt.imshow(img)
 