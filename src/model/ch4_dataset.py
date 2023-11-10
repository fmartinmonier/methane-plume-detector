import os
import sys

import numpy as np
import torch
import torchvision

class UNetTestDataset(torch.utils.data.Dataset):
    def __init__(self, scene_list, ogim_mean, ogim_std):
        super().__init__()
        self.scene_list = scene_list

        self.ogim_mean = ogim_mean 
        self.ogim_std = ogim_std

    def __len__(self):
        return len(self.scene_list)
    
    def __getitem__(self, idx):
        scene = self.scene_list[idx]

        image_width, image_height = scene["B2"].shape
        target_resolution = (512, 512)
        
        scene_date = scene["scene_date"].strftime("%Y%m%d_%H%M%S")

        channels = {
            "ndmi": np.nan_to_num(scene["ndmi"]),
            "bsi": np.nan_to_num(scene["bsi"]),
            "ndbi": np.nan_to_num(scene["ndbi"]),
            "ndvi": np.nan_to_num(scene["ndvi"]),
            "blue_channel": np.nan_to_num(scene["B2"]),
            "green_channel": np.nan_to_num(scene["B3"]),
            "red_channel": np.nan_to_num(scene["B4"]),
            "nir_channel": np.nan_to_num(scene["B8"]),
            "swir11_channel_stan" : np.nan_to_num(scene["b11_stan"]),
            "swir12_channel_stan" : np.nan_to_num(scene["b12_stan"]),
        }
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # Convert the image to a tensor
        ])
        
        image = np.stack(list(channels.values()), axis=-1).astype(np.float32)
        transformed_image = transform(image)
        
        # Some Sentinel-2 scenes have nan values due to S2 missing tiles. Those yield 0 values which we mask
        # out for the network.
        pixel_mask = torch.tensor(np.where(scene["ndmi"] == np.nan, 0, 1))

        return {
            "image": transformed_image,
            "pixel_mask": pixel_mask,
            "scene_date": scene_date,
        }

def collate_fn(batch):
    image = [item["image"] for item in batch]
    pixel_mask = [item["pixel_mask"] for item in batch]
    scene_date = [item["scene_date"] for item in batch]

    # Find the largest image size in the batch
    max_width = max([item.shape[2] for item in image])
    max_height = max([item.shape[1] for item in image])

    # Pad images to the largest size
    pad_width, pad_height = (512, 512)
    padded_image = [torch.nn.functional.pad(item, pad=(
        0, 
        pad_width - item.shape[2], 
        0, 
        pad_height - item.shape[1])) for item in image]
    padded_pixel_mask = [torch.nn.functional.pad(item, pad=(
        0, 
        pad_width - item.shape[1], 
        0, 
        pad_height - item.shape[0])) for item in pixel_mask]

    batch = {}
    batch["image"] = torch.stack(padded_image)
    batch["pixel_mask"] = torch.stack(padded_pixel_mask)
    batch["scene_date"] = scene_date

    return batch