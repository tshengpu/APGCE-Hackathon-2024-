import numpy as np
import cv2
import os

def Extract_Fault_Coordinate():
    path = r'C:\Users\chuny\OneDrive\Desktop\Geohackathon 2024\faults'
    for name in os.listdir(path):
        filename = name.split('.')[0]
        pathname = os.path.join(path, name)
        img = cv2.imread(pathname)

        # Convert to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply binary threshold to get the black pixels
        _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)

        # Convert to binary format (0 and 1)
        binary_mask = binary_mask // 255
        output_path = r'C:\Users\chuny\OneDrive\Desktop\Geohackathon 2024\faults_numpy'
        output_path = output_path + "\\" + filename + '.npy'
        # Save the binary mask as a .npy file
        np.save(output_path, binary_mask)


def Overlay_Fault():

    fault_mask_path = r'C:\Users\chuny\OneDrive\Desktop\Geohackathon 2024\faults_numpy'
    seismic_path = r'C:\Users\chuny\OneDrive\Desktop\Geohackathon 2024\seismics1'

    img = cv2.imread(os.path.join(seismic_path, "seismic-1000.png"))
    fault_mask = np.load(os.path.join(fault_mask_path, "fault-1000.npy"))
    inverted_fault_mask = cv2.bitwise_not(fault_mask * 255)

    # Convert binary image to a 3-channel image for overlay (BGR format)
    fault_mask_bgr = cv2.cvtColor(inverted_fault_mask, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(img, 0.5, fault_mask_bgr, 0.5, 0)

    cv2.imwrite("overlay_img_fault.png", overlay)


def remove_white_background(image, output_path=None):
    """
    Removes white background from an image and optionally saves the result as a PNG with transparency.

    Args:
        image_path (str): Path to the input image.
        output_path (str, optional): Path to save the output image. If None, the result is not saved.

    Returns:
        result (numpy.ndarray): The image with white background removed.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a binary mask where white pixels are detected
    _, binary_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Invert the binary mask to get the foreground
    inverted_mask = cv2.bitwise_not(binary_mask)

    # Create a new image with an alpha channel (transparency)
    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Set the alpha channel based on the inverted mask
    result[:, :, 3] = inverted_mask

    # Save the result if output_path is provided
    if output_path:
        cv2.imwrite(output_path, result)

    return result

def remove_white_space_area(image, kernel_size=10, offset=False, offsetMeasure=30):
    """
    Crops the image to the largest bounding box around non-white areas, optionally applying an offset.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the cropped image.
        kernel_size (int): Size of the kernel used for morphological operations to expand areas.
        offset (bool): Whether to apply an offset to the bounding box.
        offsetMeasure (int): Amount of offset to apply if offset is True.
    """ 
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define white color range in grayscale
    white_min = 240  # This can be adjusted based on the whiteness threshold
    white_max = 255
    
    # Create a binary mask where non-white areas are 1
    _, binary_mask = cv2.threshold(gray, white_min, white_max, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations to expand the non-white areas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    expanded_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the expanded binary mask
    contours, _ = cv2.findContours(expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding box for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Apply offset if specified
    if offset:
        x -= offsetMeasure
        y -= offsetMeasure
        w += 2 * offsetMeasure
        h += 2 * offsetMeasure
    
    # Ensure the bounding box coordinates are within image boundaries
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    # Crop the image based on the adjusted bounding box
    cropped_image = image[y:y + h, x:x + w]

    return cropped_image

# Load your image
seismic_path = r'C:\Users\chuny\OneDrive\Desktop\Geohackathon 2024\seismics1'
img = cv2.imread(os.path.join(seismic_path, "seismic-1000.png"))
    
# Remove padding
# result = remove_non_uniform_padding(img)
result = remove_white_space_area(img)
    
# Display the result
# cv2.imshow("Cropped Image", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
# Optionally save the result
cv2.imwrite(os.path.join(seismic_path, "seismic-1000_cropped.png"), result)

Overlay_Fault()
Extract_Fault_Coordinate()
