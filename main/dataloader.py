from typing import List
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from config import DatasetConfig, DataProcessingConfig


# Custom thresholding transform
class ThresholdTransform:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, tensor):
        return torch.where(tensor < self.threshold, 0, tensor)

        
class DataLoader(Dataset): 
    def __init__(self, seismic_files: List, fault_files: List, horizon_files: List, img_size = (448, 448)):
        self.RAW_SEISMIC_FOLDER = DatasetConfig.RAW_SEISMIC_FOLDER
        self.RAW_FAULT_FOLDER = DatasetConfig.RAW_FAULT_FOLDER
        self.RAW_HORIZON_FOLDER = DatasetConfig.RAW_HORIZON_FOLDER
        self.FAULT_MASK_FOLDER = DatasetConfig.FAULT_MASK_FOLDER
        self.seismic_files = seismic_files
        self.fault_files = fault_files
        self.horizon_files = horizon_files
        self.IMAGE_SIZE = img_size
        self.input_transforms = self._input_transforms()
        # self.output_transforms = self._output_transforms()
        # NOT BEING USED
        with open(os.path.join(DatasetConfig.DATA_FOLDER, 'class_names.txt'), 'r') as f:
            labels = f.readlines()
        self.label = {}
        for i, label in enumerate(labels):
            self.label[i] = label.strip()

    def _input_transforms(self):
        transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),               # Resize image to 1024x1024
            transforms.ToTensor(),                         # Convert to tensor
            ThresholdTransform(                            # Apply thresholding
                threshold=DataProcessingConfig.PIXEL_CUTOFF_THRESHOLD
            ),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.ToPILImage()                       # Convert back to PIL image
        ])
        return transform

    # def _output_transforms(self):
    #     transform = transforms.Compose([
    #         # transforms.Resize(self.IMAGE_SIZE),               # Resize image to 1024x1024
    #         transforms.Resize((self.img_size[1], self.img_size[0])),  # Resize to (Height, Width)
    #         transforms.ToTensor()                             # Convert to tensor
    #     ])
    #     return transform

    def __getitem__(self, idx):
        # print(f"Seismic file = {os.path.join(self.RAW_SEISMIC_FOLDER, self.seismic_files[idx])}")
        # print(f"Fault file = {os.path.join(self.RAW_FAULT_FOLDER, self.fault_files[idx])}")
        # print()
        
        img = Image.open(os.path.join(self.RAW_SEISMIC_FOLDER, self.seismic_files[idx]))
        img = self.input_transforms(img)
        # fault_name = '.'.join(self.img_name[idx].split('.', maxsplit = 1)[:-1]) + '.npy'
        # fault = np.load(os.path.join(self.RAW_FAULT_FOLDER, self.fault_files[idx]), allow_pickle=True).astype(np.uint8)
        # fault = cv2.resize(mask, self.img_size, cv2.INTER_AREA)
        # fault_img = Image.open(os.path.join(self.RAW_FAULT_FOLDER, self.fault_files[idx]))
        # fault = self.output_transforms(fault_img)
        # fault  = np.array(fault_img)
        # mask_name = '.'.join(self.img_name[idx].split('.', maxsplit = 1)[:-1]) + '.npy'
        mask = np.load(os.path.join(self.FAULT_MASK_FOLDER, self.fault_files[idx])).astype(np.uint8)
        mask = cv2.resize(mask, self.IMAGE_SIZE, cv2.INTER_AREA)
        
        return img, torch.from_numpy(mask.astype(np.int64))

    def __len__(self):
        return len(self.seismic_files)