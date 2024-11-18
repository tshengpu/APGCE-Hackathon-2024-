import numpy as np
import pandas as pd
from PIL import Image
import os

faults = []
seismics = []

for name in os.listdir('../../data/raw_fault'):
    if name == '.ipynb_checkpoints':
        continue
    if os.path.isfile('../../data/raw_seismic/{}'.format(name.replace("fault","seismic"))):
        faults.append(name)
        seismics.append(name.replace("fault","seismic"))

df = pd.DataFrame({
    'RAW_SEISMIC': seismics,
    'RAW_FAULT': faults
})



def sample_window_from_image(image_path, x, y, width, height, flip=False):
    """
    Samples a specific window from a PNG image based on coordinates.

    Parameters:
    - image_path (str): The file path to the PNG image.
    - x (int): The x-coordinate of the top-left corner of the window.
    - y (int): The y-coordinate of the top-left corner of the window.
    - width (int): The width of the window.
    - height (int): The height of the window.

    Returns:
    - Image: A Pillow Image object representing the sampled window.
    """
    # Open the image using Pillow
    with Image.open(image_path) as img:
        # Define the box to crop: (left, upper, right, lower)
        box = (x, y, x + width, y + height)
        
        # Crop the image
        window = img.crop(box)
        if flip:
            window = window.transpose(Image.FLIP_LEFT_RIGHT)
        
    return window


# Input values
output_count = 50
whole_width = 4166
whole_height = 2664
width = 512
height = 512


np.random.seed(42)
index = np.random.randint(len(df), high=None, size=output_count)
x_vals = np.random.randint(whole_width-width, high=None, size=output_count)
y_vals = np.random.randint(whole_height-height, high=None, size=output_count)
flip_vals = np.random.randint(2, high=None, size=output_count)
for i in range(output_count):
    for fol in ["RAW_SEISMIC","RAW_FAULT"]:
        infilepath = "../../data/{}/{}".format(fol.lower(), df.at[index[i],fol])
        x = x_vals[i]
        y = y_vals[i]
        outfilepath = "../../data/aug_{}/{}_{}_{}_{}.png".format(fol.lower(), df.at[index[i],fol].replace(".png",""), x, y, flip_vals[i])
    
        # Sample the window from the image
        sampled_window = sample_window_from_image(infilepath, x, y, width, height, flip_vals[i])
    
        # Show the sampled window
        # sampled_window.show()  # This will display the cropped window
        # break
    
    # # Optionally save the sampled window
        sampled_window.save(outfilepath)
    print(f"({i+1}/{output_count}) Sampled window saved to: {outfilepath}")