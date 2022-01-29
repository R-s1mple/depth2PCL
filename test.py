from imageio import imread
import numpy as np 
depth = imread("D:\\paper\\论文\\cloud2depth\\depth\\873_v.JPG.depth.png")[...,None].astype(np.float32)
h,w = depth.shape[:2]
print(np.min(depth))
for i in range(h):
    depth[i,0] = 0.

