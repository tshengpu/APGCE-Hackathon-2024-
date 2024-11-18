import os
import pandas as pd
import numpy as np
import copy
import torch
import loss
from torch import optim
from metrics import eval_metrics, get_epoch_acc
from dataloader import DataLoader
from cross_val import CrossVal
from torchvision import transforms
from eval import eval
from config import ModelParameters
from PIL import Image
import cv2
import math

# Import available models, you can also explore other PyTorch models
from cracknet import cracknet, CrackNet
# from unet import UNet, UNetResnet
# from segnet import SegNet, SegResNet


import os
import torch
import math
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader, TensorDataset
from dataloader import ThresholdTransform
from config import DataProcessingConfig

class Group4ModelWrapper:
    def __init__(self, filename, model = "model.pt"):
        self.filename = filename
        self.DEVICE = "cpu"
        self.model = torch.load(model, weights_only=False)
        self.model = self.model.to(self.DEVICE)
        self.batch_size = 1  # Set an appropriate batch size
        
        # Define transformations
        self.input_transforms = transforms.Compose([
            transforms.Resize((512,512)),               # Resize image to 1024x1024
            transforms.ToTensor(),                         # Convert to tensor
            ThresholdTransform(                            # Apply thresholding
                threshold=DataProcessingConfig.PIXEL_CUTOFF_THRESHOLD
            ),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.ToPILImage()                       # Convert back to PIL image
        ])
        
    def run_predict(self):
        if not os.path.isfile(self.filename):
            print(f"Error reading file: {self.filename}")
            return None, None
        
        original_img = Image.open(self.filename)
        print(f"Ori={original_img.size}")
        width, height = original_img.size
        
        new_width = math.ceil(width / 512) * 512
        new_height = math.ceil(height / 512) * 512
        padded_img = Image.new(original_img.mode, (new_width, new_height), (255, 255, 255))
        padded_img.paste(original_img, (0, 0))

        print(f"Padded={padded_img.size}")
        
        img_patches = []
        for i in range(0, new_height, 512):
            for j in range(0, new_width, 512):
                cropped_img = self.crop(padded_img, j, i)
                transformed_img = self.input_transforms(cropped_img)
                img_patches.append(transformed_img)

        print(f"len patches={len(img_patches)}")
        
        img_tensor = torch.stack(img_patches, dim=0).to(self.DEVICE)
        mask_pred = self.model(img_tensor)
        
        _, predict = torch.max(mask_pred.data, 1)
        
        predict_np = predict.cpu().numpy()
        print(f"predict_np={predict_np.size}")

        # predict_np_patches = torch.unbind(predict_np, dim=0)
        # print(f"len patches after={len(predict_np_patches)}")
        
        # Reshape and reconstruct the mask
        row_count = new_height // 512
        col_count = new_width // 512
        predict_np = predict_np.reshape(row_count, col_count, 512, 512)
        full_mask = np.block([[predict_np[i, j] for j in range(col_count)] for i in range(row_count)])
        full_mask = full_mask[:height, :width]
        
        inverted_fault_mask = (255 - full_mask * 255).astype('uint8')
        fault_mask_bgr = cv2.cvtColor(inverted_fault_mask, cv2.COLOR_GRAY2BGR)
        
        return fault_mask_bgr, original_img, predict, predict_np, mask_pred


    def crop(self, img, x, y, width=512, height=512):
        """
        Crop the image based on given coordinates.
        """
        # Define the box to crop: (left, upper, right, lower)
        box = (x, y, x + width, y + height)
        window = img.crop(box)
        return window

    def overlay(self, mask, original_img, alpha=0.5):
        """
        Overlay the prediction mask on top of the original image.
        
        Args:
            mask (numpy array): The predicted mask (in BGR format).
            original_img (PIL Image): The original input image.
            alpha (float): Transparency factor (0.0 - 1.0).

        Returns:
            numpy array: The blended overlay image.
        """
        # Convert the original image to a NumPy array
        original_np = np.array(original_img)
        
        # Resize the mask if needed to match the original image dimensions
        if mask.shape[:2] != original_np.shape[:2]:
            mask = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]))
        
        # Perform the overlay using addWeighted
        overlay_image = cv2.addWeighted(original_np, alpha, mask, 1 - alpha, 0)
        return overlay_image


# Example Usage
if __name__ == "__main__":
    # Instantiate the model wrapper with the image and model file paths
    model_wrapper = Group4ModelWrapper("../data/raw_seismic/seismic-1314.png")
    
    # Run the prediction
    fault_mask_bgr, original_img, predict, predict_np, mask_pred = model_wrapper.run_predict()
    
    if fault_mask_bgr is not None and original_img is not None:
        # Overlay the mask onto the original image
        result_overlay = model_wrapper.overlay(fault_mask_bgr, original_img, alpha=0.5)
        
        # Display the resulting overlay
        # cv2.imshow("Overlay Result", result_overlay)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Save the overlay result if needed
        cv2.imwrite("overlay_output.png", result_overlay)
    else:
        print("Prediction failed.")
