import torch
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import glob
from utils.inferenceUtils import overlay_masks
import numpy as np

from scipy.ndimage import measurements

matplotlib.use('Agg')
# parameters
threshold = 0.4

trainImgs = glob.glob('Droplets/Test/hydrophobic/*.png')
hpiTest = glob.glob('Droplets/Test/hydrophilic/*.png')
hpoTest = glob.glob('Droplets/Test/hydrophobic/*.png')

model = torch.load('D:\deeplab models\HPo trained (with pretraining)/BestTrainWeights.pt')
model.eval()

# Read image
frameIdx = 1
img0 = cv.imread(hpoTest[0])
img0 = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
img0gray = cv.cvtColor(img0, cv.COLOR_RGB2GRAY)
img = img0.transpose(2, 0, 1).reshape(1, 3, 500, 500)
# test = cv.cvtColor(test0, cv.COLOR_BGR2RGB).transpose(2, 0, 1)
with torch.no_grad():
    mask = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)
mask2d = mask['out'].cpu().detach().numpy()[0][0]
mask2d[mask2d > threshold] = 1
mask2d[mask2d < threshold] = 0
boolmask2d = mask2d == 1
labeled_pred, num_droplets = measurements.label(mask2d)
np.where

# Overlay image
maskedImage = overlay_masks(img0gray, [boolmask2d], colors=[np.array([239, 95, 58, 255])/255.0], mask_alpha=0.3)
maskedImage.save('C:/Users/user/PycharmProjects/DeeplabV3_droplet/ExtractedData/maskedFrames/allDroplets_frame'+str(frameIdx)+'.png')
