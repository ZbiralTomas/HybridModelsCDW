import os
import cv2


# Access an image for all classes and apply the predicted masks
def process_directory(data_dir):
    for material_class in os.listdir(data_dir):
        if material_class == '.DS_Store':
            continue
        class_path = os.path.join(data_dir, material_class)
        images_path = os.path.join(class_path, 'images')
        predicted_masks_path = os.path.join(class_path, 'predicted_masks')
        masked_images_path = os.path.join(class_path, 'masked_images')

        os.makedirs(masked_images_path, exist_ok=True)

        for image_name in os.listdir(images_path):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(images_path, image_name)
                predicted_mask_path = os.path.join(predicted_masks_path, image_name)

                # Load the input image and predicted mask
                input_image = cv2.imread(image_path)
                binary_mask_resized = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

                # Apply the mask to the image
                masked_image = cv2.bitwise_and(input_image, input_image, mask=binary_mask_resized)

                # Save the masked image
                masked_image_name = f"masked_{image_name}"
                masked_image_path = os.path.join(masked_images_path, masked_image_name)
                cv2.imwrite(masked_image_path, masked_image)


data_dir = 'images/original_data'
process_directory(data_dir)

print("Masked images saved in the respective class directories.")