#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon April 1 11:57:46 2024

@author: soumensmacbookair
"""

from PIL import Image
import numpy as np
import pandas as pd
import os, sys, pathlib, shutil
import cv2

# Get all the images path
cwd = pathlib.Path.cwd()
is_structure = True

if is_structure:
    images_array = []
    is_rename = False
    for i in pathlib.Path.iterdir(pathlib.Path(cwd, "images")):
        for j in pathlib.Path.iterdir(pathlib.Path(i)):
            for k in pathlib.Path.iterdir(pathlib.Path(j)):
                for l in pathlib.Path.iterdir(pathlib.Path(k)):
                    if is_rename:
                        new_name = pathlib.Path(l.parent, l.parent.stem + "_" + l.name.split("_")[-1])
                        pathlib.Path.rename(l, new_name)
                        images_array.append(new_name)
                    else:
                        images_array.append(l)
                        
    images_array = np.array(images_array)

    # Create train validation split
    np.random.shuffle(images_array)
    num_images = images_array.__len__()
    train_images_array = images_array[0:int(num_images*0.8)]
    val_images_array = images_array[int(num_images*0.8):]

    # Create train validation folders
    is_new_dir = True
    if is_new_dir:
        pathlib.Path.mkdir(pathlib.Path(cwd, "train", "sharp"), parents=True, exist_ok=False)
        pathlib.Path.mkdir(pathlib.Path(cwd, "train", "blur"), parents=True, exist_ok=False)
        pathlib.Path.mkdir(pathlib.Path(cwd, "val", "sharp"), parents=True, exist_ok=False)
        pathlib.Path.mkdir(pathlib.Path(cwd, "val", "blur"), parents=True, exist_ok=False)

        for c,i in enumerate(train_images_array):
            shutil.copy(i, pathlib.Path(cwd, "train", "sharp"))
            print(f"Train {c} copied")
        for c,i in enumerate(val_images_array):
            shutil.copy(i, pathlib.Path(cwd, "val", "sharp"))
            print(f"Val {c} copied")

# Resize the train and val images
train_images_array = []
for i in pathlib.Path.iterdir(pathlib.Path(cwd, "train", "sharp")):
    train_images_array.append(i)
train_images_array = np.array(train_images_array)

val_images_array = []
for i in pathlib.Path.iterdir(pathlib.Path(cwd, "val", "sharp")):
    val_images_array.append(i)
val_images_array = np.array(val_images_array)

is_resize = False

if is_resize:
    for c,i in enumerate(train_images_array):
        image = Image.open(i)
        new_image = image.resize((256, 448))
        new_image_name = pathlib.Path(i.parent, "resize_" + i.name)
        new_image.save(new_image_name)
        print(f"Train {c} resized")

    for c,i in enumerate(val_images_array):
        image = Image.open(i)
        new_image = image.resize((256, 448))
        new_image_name = pathlib.Path(i.parent, "resize_" + i.name)
        new_image.save(new_image_name)
        print(f"Val {c} resized")

# Blur the images
is_blur = True

if is_blur:
    for c,i in enumerate(train_images_array):
        image = cv2.imread(str(i))
        for k, s in zip([3, 7, 11],[0.3, 1, 1.6]):
            blur_image = cv2.GaussianBlur(image, (k, k), s)
            new_image_name = pathlib.Path(i.parent.parent, "blur", f"kernel_{k}_" + i.name)
            cv2.imwrite(str(new_image_name), blur_image)
        print(f"Train {c} blurred")

    for c,i in enumerate(val_images_array):
        image = cv2.imread(str(i))
        for k, s in zip([3, 7, 11],[0.3, 1, 1.6]):
            blur_image = cv2.GaussianBlur(image, (k, k), s)
            new_image_name = pathlib.Path(i.parent.parent, "blur", f"kernel_{k}_" + i.name)
            cv2.imwrite(str(new_image_name), blur_image)
        print(f"Val {c} blurred")

# Create a CSV file
num_train_blur_images = len(train_images_array) * 3
num_val_blur_images = len(val_images_array) * 3

train_info_df = pd.DataFrame(index=range(num_train_blur_images), columns=["blur", "sharp"])
val_info_df = pd.DataFrame(index=range(num_val_blur_images), columns=["blur", "sharp"])

for i, name in enumerate(pathlib.Path.iterdir(pathlib.Path(cwd, "train", "blur"))): 
    train_info_df.iloc[i,0] = str(name)
   
    train_info_df.iloc[i,1] = str(pathlib.Path(
        name.parent.parent, "sharp", "_".join(name.name.split("_")[2:])))
 
for i, name in enumerate(pathlib.Path.iterdir(pathlib.Path(cwd, "val", "blur"))): 
    val_info_df.iloc[i,0] = str(name)
   
    val_info_df.iloc[i,1] = str(pathlib.Path(
        name.parent.parent, "sharp", "_".join(name.name.split("_")[2:])))
    
train_info_df.to_csv("train_info.csv")   
val_info_df.to_csv("val_info.csv")  

# Create test images csv file
test_images_array = []
for i in pathlib.Path.iterdir(pathlib.Path(cwd, "custom_test", "blur")):
    test_images_array.append(i)
test_images_array = np.array(test_images_array)

num_test_blur_images = len(test_images_array)

test_info_df = pd.DataFrame(index=range(num_test_blur_images), columns=["blur", "sharp"])

for i, name in enumerate(test_images_array): 
    test_info_df.iloc[i,0] = str(name)
   
    test_info_df.iloc[i,1] = str(pathlib.Path(name.parent.parent, "sharp", name.name))
  
test_info_df.to_csv("csv_inputs/test_info.csv")