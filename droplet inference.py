import torch
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import glob
import numpy as np

trainImgs = glob.glob('Droplets/Test/hydrophobic/*.png')
hpiTest = glob.glob('Droplets/Test/hydrophilic/*.png')
hpoTest = glob.glob('Droplets/Test/hydrophobic/*.png')

model = torch.load('./CFExp/BestValWeights.pt')
model.eval()

test0 = cv.imread(hpiTest[0])
test = cv.cvtColor(test0, cv.COLOR_BGR2RGB).transpose(2, 0, 1).reshape(1, 3, 500, 500)
foo = test.transpose(0,2,3,1)
with torch.no_grad():
    pred = model(torch.from_numpy(test).type(torch.cuda.FloatTensor)/255)

plt.figure()
plt.imshow(test0)
plt.figure()
plt.imshow(pred['out'].cpu().detach().numpy()[0][0] > 0.5)