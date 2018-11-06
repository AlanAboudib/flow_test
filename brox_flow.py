# Some imports
import numpy as np
from PIL import Image
import time
import pyflow
import os
import matplotlib.pyplot as plt

# path to the folder containing extracted video frames
src_pth = './data/image_seqs/video1'

# path to which flow images should be saved
dst_pth = './data/flow/video1/'

# Flow Options:

alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))


# get a list of frame names
frame_names = sorted(os.listdir(src_pth))

# count the number of flow compute operations
counter = 0

# compute overall time per video
total_time = 0

# iterate through pairs of frames
for i in range(len(frame_names) - 1):

    im1_path = os.path.join(src_pth, frame_names[i])
    im1 = np.array(Image.open(im1_path))
    im1 = im1.astype(float) / 255
    
    im2_path = os.path.join(src_pth, frame_names[i + 1])
    im2 = np.array(Image.open(im2_path))
    im2 = im2.astype(float) / 255    


    counter += 1

    s = time.time()
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,  nSORIterations, colType)

    e = time.time()

    total_time += e - s



print('Average time taken per compute operation: %.2f' % (total_time / counter))
