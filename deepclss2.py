# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 02:27:22 2023

@author: swaroop hn
"""

import numpy as np # linear algebra
import os

BASE_PATH = 'D:\image/Dataset_BUSI_with_GT'
unique_classes = []
for path in os.listdir(BASE_PATH):
    unique_classes.append(path)
print(unique_classes)

class_index = [unique_classes[1], unique_classes[0], unique_classes[2]]
for c in class_index:
    print(c, "-", class_index.index(c))
    
images = []
masks = []
labels = []
for folder in os.listdir(BASE_PATH):
    class_path = os.path.join(BASE_PATH, folder)
    for img in os.listdir(class_path):
        if "_mask" not in img:
            img_path = os.path.join(class_path, img)
            msk_path = img_path.replace(".png", "_mask.png")
            # check if mask exist
            if os.path.exists(msk_path):
                images.append(img_path)
                masks.append(msk_path)
                labels.append(folder)
                
print(len(images))

images[0]


    



