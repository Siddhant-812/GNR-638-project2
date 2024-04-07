import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np

def plot_loss_psnr(csv_path, kind, window_length=31, polyorder=2, xlabel=None):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Extract the data for loss and PSNR
    loss = df['loss']
    psnr = df['psnr']
    
    # Create a figure and axis objects for the subplot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot data points for loss with lines connecting them
    axs[0].plot(loss, marker='o', linestyle='-', markersize=3, label='Data Points')
    axs[0].plot(loss.index, loss, linestyle='-', alpha=0.3, color='gray')
    
    # Smooth the loss data using Savitzky-Golay filter
    loss_smooth = savgol_filter(loss, window_length=window_length, polyorder=polyorder)
    axs[0].plot(loss_smooth, label='Smooth Line', color='red', linewidth=2)
    
    axs[0].set_title(kind+' Loss')
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(kind+' Loss')
    axs[0].legend()
    
    # Plot data points for PSNR with lines connecting them
    axs[1].plot(psnr, marker='o', linestyle='-', markersize=3, label='Data Points')
    axs[1].plot(psnr.index, psnr, linestyle='-', alpha=0.3, color='gray')
    
    # Smooth the PSNR data using Savitzky-Golay filter
    psnr_smooth = savgol_filter(psnr, window_length=window_length, polyorder=polyorder)
    axs[1].plot(psnr_smooth, label='Smooth Line', color='red', linewidth=2)
    
    axs[1].set_title(kind+' PSNR')
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(kind+' PSNR')
    axs[1].legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.savefig(csv_path.replace(".csv","_plot.png"))

# Example usage:
# plot_loss_psnr('/raid/speech/soumen/gnr_project/outputs/train_details.csv',
#                kind="Train", window_length=1000, polyorder=5, xlabel='Optimization Step')
# plot_loss_psnr('/raid/speech/soumen/gnr_project/outputs/val_details.csv',
#                kind="Val", window_length=10, polyorder=5, xlabel='Epoch')


import matplotlib.pyplot as plt
import cv2

def plot_images(image_dict, column_titles, psnr_titles):
    num_examples = len(image_dict)
    
    fig, axs = plt.subplots(3, num_examples, figsize=(10*num_examples, 15))
    
    for i, (example, paths) in enumerate(image_dict.items()):
        blur_path, sharp_path, deblur_path = paths
        
        # Read images
        blur_img = cv2.imread(blur_path)
        sharp_img = cv2.imread(sharp_path)
        deblur_img = cv2.imread(deblur_path)
        
        # Plot blur image
        axs[0, i].imshow(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
        axs[0, i].set_title(f'{column_titles[i]} - Blur')
        axs[0, i].axis('off')
        
        # Plot sharp image
        axs[1, i].imshow(cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB))
        axs[1, i].set_title(f'{column_titles[i]} - Sharp')
        axs[1, i].axis('off')
        
        # Plot deblur image
        axs[2, i].imshow(cv2.cvtColor(deblur_img, cv2.COLOR_BGR2RGB))
        axs[2, i].set_title(f'{column_titles[i]} - Deblurred {psnr_titles[i]}')
        axs[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("outputs/test_image_result.png")

# Example usage:
image_name_1 = "02800000071.png"
psnr1 = "(PSNR: 31.46)"
image_name_2 = "00600000043.png"
psnr2 = "(PSNR: 30.79)"
a = f"custom_test/blur/{image_name_1}"
b = f"custom_test/sharp/{image_name_1}"
c = f"recon_image_test/{image_name_1}"
d = f"custom_test/blur/{image_name_2}"
e = f"custom_test/sharp/{image_name_2}"
f = f"recon_image_test/{image_name_2}"

image_dict = {
    "example_1": [a,b,c],
    "example_2": [d,e,f]
}

column_titles = [image_name_1, image_name_2]
psnr_titles = [psnr1, psnr2]

plot_images(image_dict, column_titles, psnr_titles)
