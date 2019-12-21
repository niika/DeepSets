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
 