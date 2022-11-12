from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from models.detr import build
import torchvision.transforms as T

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
num = '0089'
checkpoint = torch.load('out/seg_latenight/checkpoint'+num+'.pth', map_location=torch.device("cpu"))
#print(checkpoint.keys())
args = checkpoint["args"]
args.device="cuda:4"
model, criterion, postprocessors = build(checkpoint["args"])
model.load_state_dict(checkpoint["model"])
model.eval()

image = Image.open("datasets/carla2/carla_images/val/30.jpg")

img = transform(image).unsqueeze(0)
out = model(img)

scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
# threshold the confidence
#print(scores)
keep = scores > 0.85

from random import randrange
import cv2

masks = out["pred_masks"][keep]
tmp_img = np.zeros((masks.shape[1], masks.shape[2],3))
for mask in masks:
    color = [randrange(255),randrange(255),randrange(255)]
    for i in range(masks.shape[1]):
        for j in range(masks.shape[2]):
            if mask[i,j] > 0:
                tmp_img[i,j] = color

cv2.imwrite("out/seg"+num+".png",tmp_img)
        #print(masks.shape)
