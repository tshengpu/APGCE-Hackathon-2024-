import numpy as np
import pandas as pd
import os
import cv2

for name in os.listdir('../../data/aug_raw_fault_filter'):
    if name == ".ipynb_checkpoints":
        continue
    filename = name.split('.')[0]
    img = cv2.imread('../../data/aug_raw_fault_filter/' + name)

    # Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold to get the black pixels
    _, binary_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)

    # Convert to binary format (0 and 1)
    binary_mask = binary_mask // 255

    # Save the binary mask as a .npy file
    np.save('../../data/aug_fault_mask_filter/' + filename + '.npy', binary_mask)
    print(f"Finished file: {name}")