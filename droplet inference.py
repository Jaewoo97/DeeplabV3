import torch
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import glob
import numpy as np
from scipy.ndimage import measurements, morphology
matplotlib.use('Agg')

trainImgs = glob.glob('Droplets/Test/hydrophobic/*.png')
hpiTest = glob.glob('Droplets/Test/hydrophilic/*.png')
hpoTest = glob.glob('Droplets/Test/hydrophobic/*.png')

model = torch.load('D:\deeplab models\HPo trained (with pretraining)/BestTrainWeights.pt')
model.eval()

# Read image
frameIdx = 1
img0 = cv.imread(hpoTest[0])
img0 = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
img = img0.transpose(2, 0, 1).reshape(1, 3, 500, 500)
# test = cv.cvtColor(test0, cv.COLOR_BGR2RGB).transpose(2, 0, 1)
with torch.no_grad():
    mask = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)
mask2d = mask['out'].cpu().detach().numpy()[0][0]
mask2d[mask2d > 0.5] = 1
mask2d[mask2d < 0.5] = 0
labeled_pred, num_droplets = measurements.label(mask2d)
positions = measurements.find_objects(labeled_pred)

# Overlay image
plt.imshow(img0)
plt.imshow(mask2d, cmap='jet', alpha=0.5)
sdf
plt.savefig('C:/Users/user/PycharmProjects/DeeplabV3_droplet/ExtractedData/maskedFrames/allDroplets_frame'+str(frameIdx)+'.png')


plt.imshow(pred2d)

plt.figure()
plt.imshow(test0)
plt.figure()