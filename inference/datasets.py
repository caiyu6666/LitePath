import h5py
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import io


class PatchDataset(data.Dataset):
    def __init__(self, h5_path, transform=None, load_to_memory=False):
        """
        Dataset for accessing WSI patches stored in HDF5 format as JPEG byte streams
        Args:
            h5_path (str): Path to the HDF5 file
            transform (callable, optional): Optional transform to be applied on a sample
            load_to_memory (bool): If True, loads all data into memory for faster access
        """
        super().__init__()
        self.h5_path = h5_path
        self.transform = transform
        self.load_to_memory = load_to_memory
        
        # Open the file to read metadata
        with h5py.File(h5_path, 'r') as h5_file:
            self.num_samples = len(h5_file['patches'])            
            # Optionally load all data to memory
            if load_to_memory:
                # Load JPEG bytes and decode to images
                self.patches = []
                for i in range(self.num_samples):
                    jpeg_bytes = bytes(h5_file['patches'][i])
                    img = Image.open(io.BytesIO(jpeg_bytes))
                    self.patches.append(img)
            else:
                self.patches = None
                self.coords = None
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
            
        if self.load_to_memory:
            img = self.patches[index]
        else:
            # Read from disk on-the-fly
            with h5py.File(self.h5_path, 'r') as h5_file:
                # Decode JPEG bytes to image
                jpeg_bytes = bytes(h5_file['patches'][index])
                img = Image.open(io.BytesIO(jpeg_bytes))
        
        # Apply transform if specified
        if self.transform is not None:
            img = self.transform(img)
        return img

