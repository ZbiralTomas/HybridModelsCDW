import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import shannon_entropy


def calculate_means(image, mask):
    """
    Calculate various metrics for an irregular shape defined by a mask.

    Parameters:
    - image (np.array): The input image.
    - mask (np.array): A binary mask indicating the region of interest (ROI).

    Returns:
    - dict: Dictionary containing calculated metrics.
    """
    # Ensure the image and mask have the same dimensions
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask must have the same dimensions")

    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)

    # Apply the mask to get the pixel values within the ROI
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Extract the color channels
    red_channel = masked_image[:, :, 2][mask > 0]
    green_channel = masked_image[:, :, 1][mask > 0]
    blue_channel = masked_image[:, :, 0][mask > 0]

    # Calculate mean values
    mean_red = np.mean(red_channel)
    mean_green = np.mean(green_channel)
    mean_blue = np.mean(blue_channel)
    mean_brightness = mean_red + mean_green + mean_blue

    # Calculate mean relative values
    mean_relative_red = mean_red / mean_brightness
    mean_relative_green = mean_green / mean_brightness
    mean_relative_blue = mean_blue / mean_brightness

    # Calculate mean intensity gradient (MIG)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    mig = np.mean(gradient_magnitude[mask > 0])

    # Calculate Shannon's entropy
    entropy = shannon_entropy(gray_image[mask > 0])

    results = {
        'mean_red': mean_red,
        'mean_green': mean_green,
        'mean_blue': mean_blue,
        'mean_brightness': mean_brightness,
        'mean_relative_red': mean_relative_red,
        'mean_relative_green': mean_relative_green,
        'mean_relative_blue': mean_relative_blue,
        'mean_intensity_gradient': mig,
        'shannon_entropy': entropy
    }

    return results


def process_directory(data_dir, output_csv):
    results = []

    for material_class in os.listdir(data_dir):
        class_path = os.path.join(data_dir, material_class)
        if os.path.isdir(class_path) and material_class != '.DS_Store':
            images_path = os.path.join(class_path, 'images')
            masks_path = os.path.join(class_path, 'predicted_masks')

            for image_name in os.listdir(images_path):
                if image_name.endswith('.jpg'):
                    image_path = os.path.join(images_path, image_name)
                    mask_name = 'image_' + image_name.split('_')[1]
                    mask_path = os.path.join(masks_path, mask_name)

                    image = cv2.imread(image_path)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    if image is not None and mask is not None:
                        metrics = calculate_means(image, mask)
                        metrics.update({
                            'label': material_class,
                            'image_name': image_name,
                            'mask_name': mask_name
                        })
                        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results written to {output_csv}")


# Example usage
data_dir = 'images/augmented_data'
output_csv = 'csv_files/augmented_data_features.csv'

process_directory(data_dir, output_csv)