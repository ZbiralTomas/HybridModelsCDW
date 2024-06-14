import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import csv
from model import Unet
from PIL import Image


def calculate_iou(predicted_mask, gt_mask):
    if predicted_mask.shape != gt_mask.shape:
        raise ValueError("predicted_mask and gt_mask must have the same dimensions")

    predicted_mask_bool = predicted_mask.astype(np.bool_)
    gt_mask_bool = gt_mask.astype(np.bool_)

    intersection = np.logical_and(predicted_mask_bool, gt_mask_bool)
    union = np.logical_or(predicted_mask_bool, gt_mask_bool)

    intersection_area = np.sum(intersection)
    union_area = np.sum(union)

    if union_area == 0:
        return 0
    iou = intersection_area / union_area

    return iou


def process_directory(data_dir, model, transform):
    results = []
    total_predictions = 0
    for material_class in os.listdir(data_dir):
        if material_class == '.DS_Store':
            continue
        class_path = os.path.join(data_dir, material_class, 'images')
        total_predictions += len([name for name in os.listdir(class_path) if name.endswith('.jpg')])

    current_prediction = 0

    for material_class in os.listdir(data_dir):
        if material_class == '.DS_Store':
            continue
        class_path = os.path.join(data_dir, material_class)
        images_path = os.path.join(class_path, 'images')
        masks_path = os.path.join(class_path, 'masks')
        predicted_masks_path = os.path.join(class_path, 'predicted_masks')

        os.makedirs(predicted_masks_path, exist_ok=True)

        for image_name in os.listdir(images_path):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(images_path, image_name)
                mask_name = 'mask_' + image_name.split('_')[1]
                mask_path = os.path.join(masks_path, mask_name)
                predicted_mask_path = os.path.join(predicted_masks_path, image_name)

                # Load and preprocess the input image
                input_image = cv2.imread(image_path)
                gt_mask = cv2.imread(mask_path)
                gt_mask_2d = gt_mask[:, :, 0]
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_tensor = transform(input_image).unsqueeze(0)

                # Make a prediction
                with torch.no_grad():
                    output_mask = model(input_tensor)

                output_mask = output_mask.squeeze(0).numpy()

                if output_mask.ndim == 3 and output_mask.shape[0] == 1:
                    output_mask = output_mask.squeeze(0)

                threshold = 0.5
                binary_mask = (output_mask > threshold).astype(np.uint8)
                binary_mask_resized = cv2.resize(binary_mask, (input_image.shape[1], input_image.shape[0]))
                blue = np.full_like(input_image, (255, 0, 0))
                red = np.full_like(input_image, (0, 0, 255))
                blend = 0.5
                img_blue = cv2.addWeighted(input_image, blend, blue, 1 - blend, 0)
                result = np.where(binary_mask_resized[:, :, None] == 0, img_blue, input_image)
                img_red = cv2.addWeighted(result, blend, red, 1 - blend, 0)
                result_2 = np.where((gt_mask != 255) & (binary_mask_resized[:, :, None] == 1), img_red, result)
                result_2 = result_2[:, :, ::-1]
                image = Image.fromarray(result_2)
                image.save(predicted_mask_path)

                try:
                    iou = calculate_iou(binary_mask_resized, gt_mask_2d)
                except ValueError as e:
                    print(f"Skipping {image_name} due to error: {e}")
                    iou = 0

                results.append([material_class, image_name, mask_name, iou])

                current_prediction += 1
                if current_prediction % 10 == 0:
                    print(f"Progress: {current_prediction}/{total_predictions} predictions done")

    return results

# Define the transform to apply to the input image before feeding it to the model
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL Image
    transforms.Resize((256, 256)),  # Resize the image to match the input size expected by the model
    transforms.ToTensor()  # Convert the image to a PyTorch tensor
])

# Load the saved model
model = Unet()  # Initialize your model
model.load_state_dict(torch.load('saved_models_new_new/model_epoch_21.pth'))
model.eval()  # Set the model to evaluation mode

