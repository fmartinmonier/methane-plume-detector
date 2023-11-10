import os
import sys
import numpy as np
from scipy import ndimage


def change_resolution(plume, target_resolution):
    base_resolution = plume.shape
    resampling_factor = (
        target_resolution[0] / base_resolution[0],
        target_resolution[1] / base_resolution[1]
    )
    resampled_plume = ndimage.zoom(plume, resampling_factor, order=0) # order=0 means nearest neighbor interpolation. Higher order is spline
    return resampled_plume

def antialiasing_filter(scene, channels, sigma=0.5):
    filtered_scene = scene
    for c in channels:
        filtered_scene[c] = ndimage.gaussian_filter(scene[c], sigma=sigma)
    
    return filtered_scene

def standardize(raw):
    # Calculate mean and standard deviation of the array
    mean_val = np.nanmean(raw)
    std_dev = np.nanstd(raw)
    
    # Standardize array so it has 0 mean and 1 standard deviation
    standardized = (raw - mean_val) / std_dev
    
    return standardized