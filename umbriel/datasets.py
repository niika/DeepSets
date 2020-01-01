#!/usr/bin/env python3
from __future__ import print_function, division
import os
import torch
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
from utils import createOcclusion
import random 
import re
def createArtifactMask(img):
    i,m1 = createShadowMask(img)
    i,m2 = createSpecularityMask(i)
    return i, np.array(m1+m2)

def createShadowMask(img):
    img = img.astype(np.float)
    h,w,_ = np.shape(img)
    mask = np.ones((h,w), np.float)
    mask = mask.astype(np.float)
    
    shadowScalar = np.random.rand()*(0.9-0.15) + 0.15
    poly = createConvexPolygon(h,w,7,3)
    mask = cv2.fillPoly(mask,poly,shadowScalar)
    
    #mask = cv2.circle(mask,(50,50),100,shadowScalar,-1)
    mask3 = cv2.merge((mask,mask,mask))
    img = cv2.multiply(img,mask3)
       
    _,mask = cv2.threshold(mask,0.999,255,cv2.THRESH_BINARY_INV)

    return img.astype(np.uint8), mask.astype(np.uint8)

def createSpecularityMask(img):
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
    mask3 = cv2.merge((mask,mask,mask))
    img = cv2.add(img, mask3)
    _,img = cv2.threshold(img,255,255,cv2.THRESH_TRUNC)
        
    _,mask = cv2.threshold(mask,5,255,cv2.THRESH_BINARY)

    return img.astype(np.uint8), mask.astype(np.uint8)

def createOcclusionMask(img,occlusion):
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
    w,h,_ = np.shape(img)
    offset_x = np.random.randint(-5,5)
    offset_y = np.random.randint(-5,5)
    M = np.float32([[1,0,offset_x],[0,1,offset_y]])
    dst = cv2.warpAffine(img,M,(h,w))
    return dst.astype(np.uint32)

def createPolygon(w, h):
    k = np.random.randint(3,10)
    x = np.random.randint(0,w,k)
    y = np.random.randint(0,h,k)
    pts = np.stack((x,y),axis=1)
    pts = np.array(pts, 'int32')
    return [pts]


class ArtifactsDataset(Dataset):
    """Dataset containing all sequences with artifacts"""
    
    def __init__(self, root_dir, indices, transform=None, ):
        self.root_dir = root_dir
        self.transform = transform
        self.ToPIL = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        self.indices = indices

    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        directory = os.listdir(self.root_dir)
        file = self.root_dir + directory[self.indices[idx]]
        img = cv2.imread(file)
        label = img[...,::-1]
        if self.transform:
            img = self.transform(label)
        label = self.ToPIL(label)
        label = self.ToTensor(label)

        img = torch.unsqueeze(img,0)
        label = torch.unsqueeze(label,0)
        return img, label
    
class SiameseDataset(Dataset):
    """Dataset containing pairwise images"""
    
    def __init__(self, root_dir, indices, transform=None, ):
        self.root_dir = root_dir
        self.transform = transform
        self.ToPIL = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        directory = os.listdir(self.root_dir)
        file = self.root_dir + directory[self.indices[idx]]
        img = cv2.imread(file)
        label = img[...,::-1]
        if self.transform:
            img1 = self.transform(label)
            img2 = self.transform(label)
        label = self.ToPIL(label)
        label = self.ToTensor(label)

        img = torch.cat((img1,img2),0)
        #img = torch.unsqueeze(img,0)

        label = torch.unsqueeze(label,0)
        return img, label
    
class SegmentationDataset(Dataset):
    """Dataset containing pairwise images"""
    
    def __init__(self, root_dir, indices, transform=None, ):
        self.root_dir = root_dir
        self.transform = transform
        self.ToPIL = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        directory = os.listdir(self.root_dir)
        file = self.root_dir + directory[self.indices[idx]]
        img = cv2.imread(file)
        img = img[...,::-1]
        
        img1, m1 = createArtifactMask(img)
        img1 = self.ToPIL(img1)
        img1 = self.ToTensor(img1)
        m1 = m1/255
        m1_inv = 1 -m1
        m1 = torch.Tensor(m1)
        m1_inv = torch.Tensor(m1_inv)
        m1 = torch.stack((m1,m1_inv),0)

        
        img2, m2 = createArtifactMask(img)
        img2 = self.ToPIL(img2)
        img2 = self.ToTensor(img2)
        m2 = m2/255
        m2_inv = 1 -m2
        m2 = torch.Tensor(m2)
        m2_inv = torch.Tensor(m2_inv)
        m2 = torch.stack((m2,m2_inv),0)

        img = torch.cat((img1,img2),0)
        segmentation = torch.cat((m1,m2),0)
        #img = torch.unsqueeze(img,0)
        return img, segmentation

       
def logGradient(img):
    """ Compute Gradients in the log domain.
        
        Given a numpy image (H,W,3) returns a Tensor (6,H,W) containing gradients in x- and y-direction.
    """
    img = np.float32(img)
    border = 16
    img = cv2.copyMakeBorder(img,border,border,border,border,0)
    img = img +1
    img = np.log(img)

    h,w,c = img.shape
    xDev = np.array([[-1.0,1.0,0.0]])
    yDev = np.array([[-1],[1],[0]])
    laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,-1]])
    img_x = cv2.filter2D(img,-1,xDev)
    img_y = cv2.filter2D(img,-1,yDev)
    #color_x = cv2.split(img_x)
    #color_y = cv2.split(img_y)
    img_x = transforms.ToTensor()(img_x)
    img_y = transforms.ToTensor()(img_y)
    grad = torch.cat([img_x,img_y])
    return grad

def GradientToImg(grad):
    """ Reconstructs an image given its gradients. """
    grad= grad.permute(1, 2, 0)
    grad = torch.split(grad,3,2)
    img_x = grad[0].numpy()
    img_y = grad[1].numpy()

    
    color_x = cv2.split(img_x)
    color_y = cv2.split(img_y)
    result = []
    
    h,w,c = img_x.shape
    xDev = np.array([[-1.0,1.0,0.0]])
    yDev = np.array([[-1],[1],[0]])
    laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,-1]])
    
    for i in range(3):
        I_x = color_x[i]
        I_y = color_y[i]

        I = cv2.filter2D(I_x,-1,cv2.flip(xDev,-1)) + cv2.filter2D(I_y,-1,cv2.flip(yDev,-1))
        I = cv2.dft(I,flags = cv2.DFT_COMPLEX_OUTPUT)

        # Split complex matrix into real part A and imaginary part B
        A,B = cv2.split(I)

        # Prepare Deconvolution
        S = np.zeros((h,w))
        S[0,0]=4
        S[0,1]=-1
        S[1,0]=-1
        S[0,w-1]=-1
        S[h-1,0]=-1
        S = cv2.dft(S, flags = cv2.DFT_COMPLEX_OUTPUT)
        C, D = cv2.split(S)
        # Handling Zero-values
        zeroC = C==0
        zeroD = D==0
        zeros = zeroC & zeroD
        D[zeros] = 1 
        # Complex Division:
        Z = C*C + D*D
        Real = (A*C+B*D)/Z
        Imaginary = (B*C-A*D)/Z

        # Pseudo inverse
        Real[zeros] = 0
        Imaginary[zeros] = 0

        S = cv2.merge([Real,Imaginary])
        S = cv2.dft(S,flags = cv2.DFT_INVERSE+cv2.DFT_REAL_OUTPUT+cv2.DFT_SCALE)

        #S = np.exp(S-1)

        result.append(S)

    for i in range(3):
        i_min = np.min(result[i])
        i_max = np.max(result[i])
        result[i] = 255*(result[i]-i_min)/(i_max-i_min)
    S = cv2.merge(result)
    return S


class GradientDataset(Dataset):
    """Dataset containing all sequences with artifacts
    
    Generates two distorted images as input data. The data is projected into the log-domain.  
    Horizontal and vertical gradients are then computed for each color channel
    
    """
    
    def __init__(self, root_dir, indices, transform=None, ):
        self.root_dir = root_dir
        self.transform = transform
        self.ToPIL = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        self.indices = indices

    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        directory = os.listdir(self.root_dir)
        file = self.root_dir + directory[self.indices[idx]]
        img = cv2.imread(file)
        label = img[...,::-1]
        if self.transform:
            img1 = self.transform(label)
            img2 = self.transform(label)
            
        img1 = logGradient(img1)
        img2 = logGradient(img2)
        label = logGradient(label)

        data = torch.cat([img1,img2],0)
        #label = torch.unsqueeze(label,0)
        return data, label
    

class GradientDatasetOcc(Dataset):
    """Dataset containing all sequences with artifacts
    
    Generates three distorted images as input data. The data is projected into the log-domain.  
    Horizontal and vertical gradients are then computed for each color channel
    
    """
    
    def __init__(self, root_dir, indices, transform=None, ):
        self.root_dir = root_dir
        self.transform = transform
        self.ToPIL = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        self.indices = indices

    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        directory = os.listdir(self.root_dir)
        file = self.root_dir + directory[self.indices[idx]]
        img = cv2.imread(file)
        label = img[...,::-1]
        if self.transform:
            img1 = self.transform(label)
            occ_file1 = self.root_dir + directory[np.random.randint(0,len(directory))]
            occ1 = cv2.imread(occ_file1)
            occ1 = occ1[...,::-1]
            img1 = createOcclusion(img1,occ1)
            
            img2 = self.transform(label)
            occ_file2 = self.root_dir + directory[np.random.randint(0,len(directory))]
            occ2 = cv2.imread(occ_file2)
            occ2 = occ2[...,::-1]
            img2 = createOcclusion(img2,occ2)
            
            img3 = self.transform(label)
            occ_file3 = self.root_dir + directory[np.random.randint(0,len(directory))]
            occ3 = cv2.imread(occ_file3)
            occ3 = occ3[...,::-1]
            img3 = createOcclusion(img3,occ3)
            
        img1 = logGradient(img1)
        img2 = logGradient(img2)
        img3 = logGradient(img3)

        label = logGradient(label)

        data = torch.cat([img1,img2,img3],0)
        #label = torch.unsqueeze(label,0)
        return data, label

    
class DatasetOcc(Dataset):
    """Dataset containing all sequences with artifacts
    
    Generates three distorted images as input data.
    
    """
    
    def __init__(self, root_dir, indices, transform=None, ):
        self.root_dir = root_dir
        self.transform = transform
        self.ToPIL = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        self.indices = indices

    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        directory = os.listdir(self.root_dir)
        file = self.root_dir + directory[self.indices[idx]]
        img = cv2.imread(file)
        label = img[...,::-1]
        if self.transform:
            img1 = self.transform(label)
            occ_file1 = self.root_dir + directory[np.random.randint(0,len(directory))]
            occ1 = cv2.imread(occ_file1)
            occ1 = occ1[...,::-1]
            img1 = createOcclusion(img1,occ1)
            
            img2 = self.transform(label)
            occ_file2 = self.root_dir + directory[np.random.randint(0,len(directory))]
            occ2 = cv2.imread(occ_file2)
            occ2 = occ2[...,::-1]
            img2 = createOcclusion(img2,occ2)
            
            img3 = self.transform(label)
            occ_file3 = self.root_dir + directory[np.random.randint(0,len(directory))]
            occ3 = cv2.imread(occ_file3)
            occ3 = occ3[...,::-1]
            img3 = createOcclusion(img3,occ3)
        
        
        imgs = [img1,img2,img3]
        imgs = [transforms.ToTensor()(img) for img in imgs]

        
        data = torch.cat(imgs)
        #data = torch.unsqueeze(data,0)
        
        label = label.astype(np.uint8)
        label = transforms.ToTensor()(label)
        #label = torch.unsqueeze(label,0)
        return data, label
    
    
class RealData(Dataset):
    """Dataset containing all sequences with artifacts
    
    Generates three distorted images as input data.
    
    """
    
    def __init__(self, root_dir, indices, transform=None, ):
        self.root_dir = root_dir
        self.transform = transform
        self.ToPIL = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        self.indices = indices

    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):

        directory = os.listdir(self.root_dir)
        subFolder = self.root_dir + str(directory[self.indices[idx]]) + "/results/"
        imgFiles = os.listdir(subFolder)
        imgFiles.remove("img0.png")
        f = lambda x: subFolder + x 
        imgFiles = list(map(f,random.sample(imgFiles, 3)))
        imgs = [cv2.imread(x) for x in imgFiles]
        

            
        
        imgs = [img[...,::-1]- np.zeros_like(img) for img in imgs]
        imgs = [ cv2.resize(img, (0,0), fx=0.5, fy=0.5) for img in imgs]
        label = cv2.imread(subFolder + "img0.png")
        label = label[...,::-1]- np.zeros_like(label)
        label = cv2.resize(label, (0,0), fx=0.5, fy=0.5)
        
        H,W,C = imgs[0].shape
        if H<W:
            label = np.rot90(label)
            label -= np.zeros_like(label)
            imgs = [np.rot90(img) for img in imgs]- np.zeros_like(label)
        
        flip = np.random.randint(-1,3)
        if flip < 2:
            label = cv2.flip(label,flip)- np.zeros_like(label)
            imgs = [cv2.flip(img,flip) for img in imgs]- np.zeros_like(label)

        imgs = [transforms.ToTensor()(img) for img in imgs]
        data = torch.cat(imgs)
        #data = torch.unsqueeze(data,0)
        
        label = label.astype(np.uint8)
        label = transforms.ToTensor()(label)
        #label = torch.unsqueeze(label,0)
        return data, label
    
    
class SimulatedData(Dataset):
    """Dataset containing all sequences with artifacts
    
    Generates three distorted images as input data.
    
    """
    
    def __init__(self, root_dir, indices, crop=False, crop_size=50, transform=None, n_images=3,resize=False):
        self.root_dir = root_dir
        self.transform = transform
        self.ToPIL = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        self.indices = indices
        self.crop = crop
        self.crop_size = crop_size
        self.n_images = n_images
        self.resize=resize
        
        files = os.listdir(root_dir)
        match = lambda x: len(re.findall("img_\d+_\d.jpg", x))== 1
        cut_string = lambda x: eval(re.sub("_.*","",re.sub("img_","",x)))

        files = list(filter(match,files))
        files = list(map(cut_string,files))


        first,last = min(files),max(files)
        self.offset = first
        self.last = last
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):        
        
        idx = self.indices[idx]
        count = 0
        img_files = None
        imgs = None
        label = None
        while True:
            nrs = np.random.choice(range(1,10),self.n_images,False).tolist()
            img_files = [self.root_dir +  "img_" +str(idx)+ "_" + str(nr) + ".jpg" for nr in nrs]
            exists = all([os.path.isfile(img_file) for img_file in img_files])
            count+=1
            try:
                imgs = [cv2.imread(file) for file in img_files]
                imgs = [img[...,::-1]- np.zeros_like(img) for img in imgs]
                #imgs = [ cv2.resize(img, dsize=(250,250)) for img in imgs]

                label_file = self.root_dir + "books/img " + "("+str(idx - 1)+").jpg"
                label = cv2.imread(label_file)
                label = label[...,::-1]- np.zeros_like(label)
                break

            except:
                idx = np.random.randint(len(self.indices))
                idx = self.indices[idx]

        
        
        if self.resize:
            label = cv2.resize(label, dsize=(256,256))
            imgs = [ cv2.resize(img, dsize=(256,256)) for img in imgs]
        
        H,W,C = imgs[0].shape
        if H<W:
            label = np.rot90(label)
            label -= np.zeros_like(label)
            imgs = [np.rot90(img) for img in imgs]- np.zeros_like(label)
        
        flip = np.random.randint(-1,3)
        if flip < 2:
            label = cv2.flip(label,flip)- np.zeros_like(label)
            imgs = [cv2.flip(img,flip) for img in imgs]- np.zeros_like(label)

        if self.crop:
            i = np.random.randint(0,H-self.crop_size)
            j = np.random.randint(0,W-self.crop_size)
            label = label[i:i+self.crop_size,j:j+self.crop_size]
            imgs = [img[i:i+self.crop_size,j:j+self.crop_size] for img in imgs]
        
        imgs = [transforms.ToTensor()(img) for img in imgs]
        data = torch.cat(imgs)
        #data = torch.unsqueeze(data,0)
        
        label = label.astype(np.uint8)
        label = transforms.ToTensor()(label)
        #label = torch.unsqueeze(label,0)
        return data, label
    
def Gradient(img):
    img = np.float32(img)
    border = 4
    img = cv2.copyMakeBorder(img,border,border,border,border,0)
    #img = img +1
    #img = np.log(img)

    h,w,c = img.shape
    xDev = np.array([[-1.0,1.0,0.0]])
    yDev = np.array([[-1],[1],[0]])
    laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,-1]])
    img_x = cv2.filter2D(img,-1,xDev)
    img_y = cv2.filter2D(img,-1,yDev)
    #color_x = cv2.split(img_x)
    #color_y = cv2.split(img_y)
    img_x = transforms.ToTensor()(img_x)
    img_y = transforms.ToTensor()(img_y)
    grad = torch.cat([img_x,img_y])
    return grad

def GradientToImg(grad):
    grad= grad.permute(1, 2, 0)
    grad = torch.split(grad,3,2)
    img_x = grad[0].numpy()
    img_y = grad[1].numpy()

    
    color_x = cv2.split(img_x)
    color_y = cv2.split(img_y)
    result = []
    
    h,w,c = img_x.shape
    xDev = np.array([[-1.0,1.0,0.0]])
    yDev = np.array([[-1],[1],[0]])
    laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,-1]])
    
    for i in range(3):
        I_x = color_x[i]
        I_y = color_y[i]

        I = cv2.filter2D(I_x,-1,cv2.flip(xDev,-1)) + cv2.filter2D(I_y,-1,cv2.flip(yDev,-1))
        I = cv2.dft(I,flags = cv2.DFT_COMPLEX_OUTPUT)

        # Split complex matrix into real part A and imaginary part B
        A,B = cv2.split(I)

        # Prepare Deconvolution
        S = np.zeros((h,w))
        S[0,0]=4
        S[0,1]=-1
        S[1,0]=-1
        S[0,w-1]=-1
        S[h-1,0]=-1
        S = cv2.dft(S, flags = cv2.DFT_COMPLEX_OUTPUT)
        C, D = cv2.split(S)
        # Handling Zero-values
        zeroC = C==0
        zeroD = D==0
        zeros = zeroC & zeroD
        D[zeros] = 1 
        # Complex Division:
        Z = C*C + D*D
        Real = (A*C+B*D)/Z
        Imaginary = (B*C-A*D)/Z

        # Pseudo inverse
        Real[zeros] = 0
        Imaginary[zeros] = 0

        S = cv2.merge([Real,Imaginary])
        S = cv2.dft(S,flags = cv2.DFT_INVERSE+cv2.DFT_REAL_OUTPUT+cv2.DFT_SCALE)

        #S = np.exp(S-1)

        result.append(S)

    for i in range(3):
        i_min = np.min(result[i])
        i_max = np.max(result[i])
        result[i] = 255*(result[i]-i_min)/(i_max-i_min)
    S = cv2.merge(result)
    return S


class SimulatedDataGrad(Dataset):
    """Dataset containing all sequences with artifacts
    
    Generates three distorted images as input data.
    
    """
    
    def __init__(self, root_dir, indices, transform=None, ):
        self.root_dir = root_dir
        self.transform = transform
        self.ToPIL = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        self.indices = indices
        
        files = os.listdir(root_dir)
        match = lambda x: len(re.findall("img_\d+_\d.jpg", x))== 1
        cut_string = lambda x: eval(re.sub("_.*","",re.sub("img_","",x)))

        files = list(filter(match,files))
        files = list(map(cut_string,files))


        first,last = min(files),max(files)
        self.offset = first
        self.last = last
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):        
        
        idx = self.indices[idx]
        count = 0
        img_files = None
        imgs = None
        label = None
        while True:
            nrs = np.random.choice(range(1,10),3,False).tolist()
            img_files = [self.root_dir +  "img_" +str(idx)+ "_" + str(nr) + ".jpg" for nr in nrs]
            exists = all([os.path.isfile(img_file) for img_file in img_files])
            count+=1
            try:
                imgs = [cv2.imread(file) for file in img_files]
                imgs = [img[...,::-1]- np.zeros_like(img) for img in imgs]
                #imgs = [ cv2.resize(img, dsize=(250,250)) for img in imgs]

                label_file = self.root_dir + "books/img " + "("+str(idx - 1)+").jpg"
                label = cv2.imread(label_file)
                label = label[...,::-1]- np.zeros_like(label)
                break

            except:
                idx = np.random.randint(len(self.indices))
                idx = self.indices[idx]

        
        
        #label = cv2.resize(label, dsize=(250,250))
        
        H,W,C = imgs[0].shape
        if H<W:
            label = np.rot90(label)
            label -= np.zeros_like(label)
            imgs = [np.rot90(img) for img in imgs]- np.zeros_like(label)
        
        flip = np.random.randint(-1,3)
        if flip < 2:
            label = cv2.flip(label,flip)- np.zeros_like(label)
            imgs = [cv2.flip(img,flip) for img in imgs]- np.zeros_like(label)

        #imgs = [transforms.ToTensor()(img) for img in imgs]
        
        imgs = [Gradient(img) for img in imgs]
        data = torch.cat(imgs)
        data = (data +255)/510
        #data = torch.unsqueeze(data,0)
        
        label = label.astype(np.uint8)
        #label = transforms.ToTensor()(label)
        label = Gradient(label)
        label = (label +255)/510

        #label = torch.unsqueeze(label,0)
        return data, label
    
    
