import os
import cv2


def find_bounding_box(binary_mask):
    """
    Find the bounding box coordinates of the binary mask.

    Parameters:
    - binary_mask (np.array): A 2D numpy array representing the binary mask.

    Returns:
    - tuple: (x, y, w, h) representing the bounding box coordinates.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    x, y, w, h = cv2.boundingRect(contours[0])
    for contour in contours:
        x_, y_, w_, h_ = cv2.boundingRect(contour)
        x = min(x, x_)
        y = min(y, y_)
        w = max(w, w_)
        h = max(h, h_)

    return (x, y, w, h)


def crop_and_apply_mask(image_path, binary_mask):
    """
    Crop the image based on the bounding box of the mask and apply the mask.
    Args:
        image_path (str): Path to the input image.
        binary_mask (np.array): The binary mask.

    Returns:
        np.array: Cropped image.
        np.array: Masked image.
    """
    input_image = cv2.imread(image_path)
    masked_image = cv2.bitwise_and(input_image, input_image, mask=binary_mask)

    bbox = find_bounding_box(binary_mask)
    if bbox is not None:
        x, y, w, h = bbox
        cropped_image = masked_image[y:y+h, x:x+w]
        return cropped_image, masked_image

    return input_image, masked_image


# Access an images for all classes, apply predicted masks, and crop the image based on predicted mask
def process_directory(data_dir):
    for material_class in os.listdir(data_dir):
        if material_class == '.DS_Store':
            continue
        class_path = os.path.join(data_dir, material_class)
        images_path = os.path.join(class_path, 'images')
        predicted_masks_path = os.path.join(class_path, 'predicted_masks')
        cropped_images_path = os.path.join(class_path, 'cropped_masked_images')

        os.makedirs(cropped_images_path, exist_ok=True)

        for image_name in os.listdir(images_path):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(images_path, image_name)
                predicted_mask_path = os.path.join(predicted_masks_path, image_name)

                # Load the input image and predicted mask
                input_image = cv2.imread(image_path)
                binary_mask_resized = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

                # Apply the mask to the image
                masked_image = cv2.bitwise_and(input_image, input_image, mask=binary_mask_resized)

                # Find the bounding box coordinates
                bbox = find_bounding_box(binary_mask_resized)
                if bbox is not None:
                    x, y, w, h = bbox
                    cropped_image = masked_image[y:y+h, x:x+w]

                    # Save the cropped image
                    cropped_image_name = f"cropped_{image_name}"
                    cropped_image_path = os.path.join(cropped_images_path, cropped_image_name)
                    cv2.imwrite(cropped_image_path, cropped_image)


data_dir = 'images/original_data'
process_directory(data_dir)

print("Cropped images saved in the respective class directories.")
