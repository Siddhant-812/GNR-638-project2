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

#%% Set the device
if torch.cuda.is_available():
    device_type = "cuda"
    print("Using GPU...")
    print(f"Total # of GPU: {torch.cuda.device_count()}")
    print(f"GPU Details: {torch.cuda.get_device_properties(device=torch.device(device_type))}")
else:
    device_type = "cpu"
    print("Using CPU...")

device = torch.device(device_type)

BATCH_SIZE = 16

transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((448, 256))
    ])

train_ds = BlurDataset(csv_path="/raid/speech/soumen/gnr_project/csv_inputs/train_info.csv", transforms=transforms)
val_ds = BlurDataset(csv_path="/raid/speech/soumen/gnr_project/csv_inputs/val_info.csv", transforms=transforms)

# Create a batch
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE//2, shuffle=True, drop_last=True)

# Get Transformer model
model = Stripformer()
model = model.to(device=device)
torchinfo.summary(model, (BATCH_SIZE, 3, 256, 448))

#%% Compile the model
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001, betas=(0.9, 0.999), weight_decay=0)
loss_fn = torch.nn.L1Loss(reduction="mean")
    
#%% Create the training loop
t1 = time.time()
epoch = 10
loss_list = []
psnr_list = []
val_loss_list_total = []
val_psnr_list_total = []
NUM_TRAIN_BATCH_PER_EPOCH = 20
NUM_VAL_BATCH_PER_EPOCH = 10

for ep in range(epoch):
    train_epoch_progress_bar = tqdm(total=NUM_TRAIN_BATCH_PER_EPOCH, desc=f"Epoch {ep+1}/{epoch}")
    
    # Create a subset of train batch
    train_subset_size = NUM_TRAIN_BATCH_PER_EPOCH*BATCH_SIZE
    train_subset_indices = torch.randperm(len(train_ds))[:train_subset_size]
    subset_train_ds = torch.utils.data.Subset(train_dl.dataset, train_subset_indices.numpy().tolist())
    subset_train_dl = torch.utils.data.DataLoader(subset_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    for i, (blur_x, sharp_y) in enumerate(subset_train_dl):
        blur_x, sharp_y = blur_x.to(device=device), sharp_y.to(device=device)
        pred_sharp_y = model(blur_x.to(torch.float32))
        loss = loss_fn(pred_sharp_y, sharp_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        psnr = peak_signal_noise_ratio(sharp_y.cpu().detach().numpy(), 
                                       pred_sharp_y.cpu().detach().numpy())
        loss_list.append(loss.item())
        psnr_list.append(psnr)
        
        train_epoch_progress_bar.set_postfix({"Train Loss": loss.item(), "Train PSNR": psnr})
        train_epoch_progress_bar.update(1)
    
    train_epoch_progress_bar.close()
    print("")
    
    # Create a subset of val batch
    val_subset_size = NUM_VAL_BATCH_PER_EPOCH*(BATCH_SIZE//2)
    val_subset_indices = torch.randperm(len(val_ds))[:val_subset_size]
    subset_val_ds = torch.utils.data.Subset(val_dl.dataset, val_subset_indices.numpy().tolist())
    subset_val_dl = torch.utils.data.DataLoader(subset_val_ds, batch_size=BATCH_SIZE//2, shuffle=True, drop_last=True)

    val_loss_list = []
    val_psnr_list = []
    val_progress_bar = tqdm(total=NUM_VAL_BATCH_PER_EPOCH)
    
    for j, (val_blur_x, val_sharp_y) in enumerate(subset_val_dl):
        val_blur_x, val_sharp_y = val_blur_x.to(device=device), val_sharp_y.to(device=device)
        val_pred_sharp_y = model(val_blur_x.to(torch.float32))
        
        val_loss = loss_fn(val_pred_sharp_y, val_sharp_y)
        val_loss_list.append(val_loss.item())
        val_psnr = peak_signal_noise_ratio(val_sharp_y.cpu().detach().numpy(), 
                                        val_pred_sharp_y.cpu().detach().numpy())
        val_psnr_list.append(val_psnr) 
        val_progress_bar.update(1)  
    
    val_progress_bar.close()
                
    val_loss_batch = sum(val_loss_list)/len(val_loss_list)
    val_psnr_batch = sum(val_psnr_list)/len(val_psnr_list)
    val_loss_list_total.append(val_loss_batch)
    val_psnr_list_total.append(val_psnr_batch)
    
    print(f"Epoch: {ep+1}/{epoch}, " +  
        f"Val Loss: {val_loss_batch:.4f}, " + 
        f"Val PSNR: {val_psnr_batch:.4f}")

    torch.save(model, f"checkpoints/model_epoch_{ep}.pth")    

train_details_df = pd.DataFrame(np.array([loss_list, psnr_list]).T, columns=["loss","psnr"])
train_details_df.to_csv("outputs/train_details.csv")
val_details_df = pd.DataFrame(np.array([val_loss_list_total, val_psnr_list_total]).T, columns=["loss","psnr"])
val_details_df.to_csv("outputs/val_details.csv")

t2 = time.time()
print(f"Time taken on {device_type}: {(t2-t1):.5f} sec")
print("END")