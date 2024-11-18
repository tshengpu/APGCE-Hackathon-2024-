import numpy as np
import pandas as pd
import os
import cv2

count = 0
for name in os.listdir('../../data/aug_fault_mask_filter'):
    if name == ".ipynb_checkpoints":
        continue
    code = name.replace("fault-","").replace(".npy", "")

    fault_mask = np.load(f'../../data/aug_fault_mask_filter/{name}')
    if np.max(fault_mask) >= 1:
        os.popen(f'cp ../../data/aug_fault_mask_filter/fault-{code}.npy ../../data/aug_fault_mask_filter_hasfault/fault-{code}.npy') 
        os.popen(f'cp ../../data/aug_raw_seismic/seismic-{code}.png ../../data/aug_raw_seismic_hasfault/seismic-{code}.png') 
        os.popen(f'cp ../../data/aug_raw_fault_filter/fault-{code}.png ../../data/aug_raw_fault_filter_hasfault/fault-{code}.png') 
        print(f"Found file: {name}")
        count += 1
print(f"Found {count} file(s) containing faults.")
