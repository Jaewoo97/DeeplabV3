import torch
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import glob
import numpy as np

model = torch.load('./CFExp/BestValWeights.pt')
model.eval()

trainImgs = glob.glob('Droplets/TestImages/hydrophobic/*.png')
hpiTest = glob.glob('Droplets/TestImages/hydrophobic/*.png')
hpoTest = glob.glob('Droplets/TestImages/hydrophobic/*.png')

test0 = cv.imread(hpiTest[0])
test = cv.cvtColor(test0, cv.COLOR_BGR2RGB).transpose(2, 0, 1).reshape(1, 3, 500, 500)
with torch.no_grad():
    pred = model(torch.from_numpy(test).type(torch.cuda.FloatTensor)/255)

plt.figure()
plt.imshow(test0)
plt.figure()
plt.imshow(pred['out'].cpu().detach().numpy()[0][0] > 0.5)