import os, cv2, time, sys
import numpy as np
import matplotlib.pyplot as plt
import torch, torchinfo
from PIL import Image
import torchvision
from typing import Tuple, List, Union
sys.path.append("/raid/speech/soumen/gnr_project")
from blur_dataset import BlurDataset

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

BATCH_SIZE = 32

train_ds = BlurDataset(csv_path="/raid/speech/soumen/gnr_project/train_info.csv", transforms=None)
val_ds = BlurDataset(csv_path="/raid/speech/soumen/gnr_project/val_info.csv", transforms=None)

# Create a batch
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

class DoubleConv(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(DoubleConv, self).__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class DownBlock(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = torch.nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

    
class UpBlock(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(UpBlock, self).__init__()
        self.up_sample = torch.nn.ConvTranspose2d(in_channels-out_channels, 
                                                  in_channels-out_channels, kernel_size=2, stride=2)        
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class UNet(torch.nn.Module):
    
    def __init__(self, out_classes=3):
        
        super(UNet, self).__init__()
        
        # Downsampling Path
        mult = 2
        self.down_conv1 = DownBlock(3, 8*mult)
        self.down_conv2 = DownBlock(8*mult, 16*mult)
        self.down_conv3 = DownBlock(16*mult, 32*mult)
        self.down_conv4 = DownBlock(32*mult, 64*mult)
        
        # Bottleneck
        self.double_conv = DoubleConv(64*mult, 128*mult)
        
        # Upsampling Path
        self.up_conv4 = UpBlock(64*mult + 128*mult, 64*mult)
        self.up_conv3 = UpBlock(32*mult + 64*mult, 32*mult)
        self.up_conv2 = UpBlock(16*mult + 32*mult, 16*mult)
        self.up_conv1 = UpBlock(8*mult + 16*mult, 8*mult)
        
        # Final Convolution
        self.conv_last = torch.nn.Conv2d(8*mult, out_classes, kernel_size=1)

    def forward(self, x):
        
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x) 
        x, skip3_out = self.down_conv3(x) 
        x, skip4_out = self.down_conv4(x) 
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x

class AutoEncoder(torch.nn.Module):
     
    def __init__(self, out_classes=3):

        super(AutoEncoder, self).__init__()
        self.cae1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(3, 3, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(3, 3, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(3, 3, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(3, 3, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.unet = UNet()
        
        self.cae2 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(3, 3, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(3, 3, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(3, 3, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(3, 3, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=False)
        )
        
    def forward(self, x):
        
        assert x.shape.__len__() == 4, "input_x must be of rank 4"
        
        x = self.cae1(x)
        x = self.unet(x)
        x = self.cae2(x)
        
        return x

# Get AE model
model = AutoEncoder()
# model = torch.nn.DataParallel(model, device_ids=[0, 1])
model = model.to(device=device)
torchinfo.summary(model, (BATCH_SIZE, 3, 256, 448))

#%% Compile the model
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
loss_fn = torch.nn.MSELoss(reduction="mean")

def calc_psnr(orig_image: torch.tensor,
             recon_image: torch.tensor,
             is_sum: bool = False) -> float:
    
    max_intensity = recon_image.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
    assert max_intensity.shape[0] == BATCH_SIZE, "something went wrong!"
    assert len(max_intensity.shape) == 1, "something went wrong!"
    
    loss = torch.zeros(size=(BATCH_SIZE,)).to(device=device)
    for i in range(BATCH_SIZE):
        loss[i] = loss_fn(orig_image[i], recon_image[i])
        
    psnr = 10 * torch.log10((255**2)/loss)
    
    if is_sum:
        return psnr.sum().item()
    else:
        return psnr.mean().item()
    
#%% Create the training loop
t1 = time.time()
epoch = 15
loss_list = []
psnr_list = []
for ep in range(epoch):
    for i, (blur_x, sharp_y) in enumerate(train_dl):
        blur_x, sharp_y = blur_x.to(device=device), sharp_y.to(device=device)
        pred_sharp_y = model(blur_x.to(torch.float32))
        loss = loss_fn(pred_sharp_y, sharp_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        psnr = calc_psnr(orig_image=sharp_y, recon_image=pred_sharp_y)
        loss_list.append(loss)
        psnr_list.append(psnr)
        print(f"Epoch: {ep+1}/{epoch}, Batch: {i+1}/{int(train_ds.__len__()/BATCH_SIZE)}, Loss: {loss:.2f}, PSNR: {psnr:.2f}")
    
    print(f"Epoch: {ep+1}/{epoch}, Last Batch Loss: {loss:.2f}, Last Batch psnr: {psnr:.2f}")
    torch.save(model, "model.pth")

t2 = time.time()
print(f"Time taken on {device_type}: {(t2-t1):.5f} sec")
print("END")