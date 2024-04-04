import os, cv2, time, sys
import numpy as np
import matplotlib.pyplot as plt
import torch, torchinfo
from PIL import Image
import torchvision
from typing import Tuple, List, Union
sys.path.append("/raid/speech/soumen/gnr_project")
from blur_dataset import BlurDataset
import pandas as pd
from Stripformer_arch import Stripformer
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
from PIL import Image
import pathlib

def get_numpy_tensor_of_image(image):
    img_tensor = image.squeeze(0)
    img_np = img_tensor.detach().numpy().transpose((1, 2, 0))
    return img_np
    
def plot_recon_image(csv_path, model_path, index):

    transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 448))
    ])
    ds = BlurDataset(csv_path=csv_path, transforms=transforms)
    
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    (blur_image, sharp_image, blur_image_name, sharp_image_name) = ds[index]
    pred_sharp_image = model(blur_image.unsqueeze(0))
    psnr = peak_signal_noise_ratio(sharp_image.detach().numpy(), pred_sharp_image.squeeze(0).detach().numpy())
    
    figname = pathlib.Path(f"/raid/speech/soumen/gnr_project/recon_image/{blur_image_name.split('/')[-1]}")
    
    # Resize the images
    pred_sharp_image_resized = torchvision.transforms.Resize((256, 448))(pred_sharp_image)
    
    # Normalize the image data to the range [0, 1]
    pred_sharp_image_norm = (pred_sharp_image_resized - pred_sharp_image_resized.min()) / (pred_sharp_image_resized.max() - pred_sharp_image_resized.min())
 
    # Save the resized images
    image_array = get_numpy_tensor_of_image(pred_sharp_image_norm)
    image_array = (image_array*255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save(str(figname))
    
    return psnr

csv_path = "/raid/speech/soumen/gnr_project/csv_inputs/test_info.csv"
model_path = "/raid/speech/soumen/gnr_project/checkpoints/model_epoch_9.pth"

psnr_list = []
for i in range(150):
    psnr_list.append(plot_recon_image(csv_path, model_path, index=i))
    print(f"test {i} done")

print(sum(psnr_list)/len(psnr_list))
print("END")