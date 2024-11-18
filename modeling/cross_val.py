import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from dataloader import DataLoader
from config import ModelParameters, DataProcessingConfig

class CrossVal(Dataset):
    def __init__(self, df, k_fold):
        self.files_df = df.dropna().reset_index(drop=True).copy()
        self.k_fold = k_fold
        self.batch_size = ModelParameters.BATCH_SIZE
        self.files_df["Fold"] = self.files_df.index % k_fold
    
    def __getitem__(self, idx):
        res = {}
        if (idx >= self.k_fold or idx < 0):
            print("Invalid idx: kFold is {}".format(self.k_fold))
            return -1
        df_train = self.files_df[self.files_df["Fold"] != idx].copy()
        df_val  = self.files_df[self.files_df["Fold"] == idx].copy()
        df_train.reset_index(drop=True, inplace=True)
        df_val.reset_index(drop=True, inplace=True)
        res["train"] = torch.utils.data.DataLoader(DataLoader(
            seismic_files = df_train['RAW_SEISMIC'],
            fault_files   = df_train['RAW_FAULT'],
            horizon_files = df_train["RAW_HORIZON"],
            img_size = DataProcessingConfig.IMAGE_SIZE
        ), batch_size = self.batch_size, shuffle = True, num_workers = 8, drop_last = False)
        res["val"] = torch.utils.data.DataLoader(DataLoader(
            seismic_files = df_val['RAW_SEISMIC'],
            fault_files   = df_val['RAW_FAULT'],
            horizon_files = df_val["RAW_HORIZON"],
            img_size = DataProcessingConfig.IMAGE_SIZE
        ), batch_size = self.batch_size, shuffle = True, num_workers = 8, drop_last = False)
        return res
        
    def __len__(self, idx):
        return len(self.files_df[self.files_df["Fold"] != idx]['RAW_SEISMIC'])
