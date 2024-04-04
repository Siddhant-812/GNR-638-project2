import os, sys, copy, cv2, pathlib, time, torch, torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Union

#%% Implement the custom Dataset class
class BlurDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        csv_path: str,
        transforms: torchvision.transforms = None
    ) -> None:

        super(BlurDataset, self).__init__()
        self.csv_path = csv_path
        self.info_df = pd.read_csv(self.csv_path)
        self.transforms = transforms

    def __len__(
        self
    ) -> int:

        return self.info_df.shape[0]

    @staticmethod
    def transform_features(
        data_x: Image.Image
    ) -> torch.tensor:

        x = torchvision.transforms.PILToTensor()(data_x)
        x = x.to(dtype=torch.float32)
        x = x/255.0

        return x

    def __getitem__(
        self,
        index: int,
        is_transform: bool = True
    ) -> Tuple[torch.tensor, torch.tensor]:

        blur_image_name = self.info_df.iloc[index,1]
        sharp_image_name = self.info_df.iloc[index,2]
 
        blur_image = Image.open(blur_image_name)
        sharp_image = Image.open(sharp_image_name)
            
        orig_blur_image = copy.deepcopy(blur_image)
        orig_sharp_image = copy.deepcopy(sharp_image)

        if self.transforms:         
            blur_image = self.transforms(blur_image)
            sharp_image = self.transforms(sharp_image)

        blur_image = BlurDataset.transform_features(blur_image)
        orig_blur_image = BlurDataset.transform_features(orig_blur_image)
        sharp_image = BlurDataset.transform_features(sharp_image)
        orig_sharp_image = BlurDataset.transform_features(orig_sharp_image)

        if is_transform:
            return blur_image, sharp_image, blur_image_name, sharp_image_name
        else:
            return orig_blur_image, orig_sharp_image, blur_image_name, sharp_image_name

    def display_sample(
        self
    ) -> None:

        random_indices = np.random.choice(np.arange(self.__len__()), size=1, replace=False)   
        index = random_indices[0]
        
        blur_image, sharp_image = self.__getitem__(index, is_transform=False)
        blur_image_name = self.info_df.iloc[index,1]
        sharp_image_name = self.info_df.iloc[index,2]

        sharp_image = sharp_image.transpose(0, 2).transpose(0, 1).to(torch.int64)
        blur_image = blur_image.transpose(0, 2).transpose(0, 1).to(torch.int64)

        # Create a figure with a 1x2 grid of subplots
        fig, axes = plt.subplots(1, 2)

        # Plot the first image on the first subplot
        axes[0].imshow(sharp_image)
        axes[0].set_title(pathlib.Path(sharp_image_name).name)

        # Plot the second image on the second subplot
        axes[1].imshow(blur_image)
        axes[1].set_title(pathlib.Path(blur_image_name).name)

        plt.tight_layout()
        plt.show()
        plt.savefig(pathlib.Path(pathlib.Path(self.csv_path).parent, str(index) + ".png"))

if __name__ == "__main__":
    
    BATCH_SIZE = 32

    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((256, 448))
    # ])

    train_ds = BlurDataset(csv_path="/raid/speech/soumen/gnr_project/train_info.csv", transforms=None)
    val_ds = BlurDataset(csv_path="/raid/speech/soumen/gnr_project/val_info.csv", transforms=None)

    # # Plot a sample of the blur dataset
    # train_ds.display_sample()
    # val_ds.display_sample()

    # Create a batch
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("END")