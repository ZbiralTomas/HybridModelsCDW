import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from unet_model import Unet  # Ensure this import path is correct


def predict_mask(model, image_path, device):
    """
    Predict the mask for the given image using the U-Net model.
    Args:
        model (torch.nn.Module): The U-Net model.
        image_path (str): Path to the input image.
        device (torch.device): The device to run the model on.

    Returns:
        np.array: The predicted mask.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = np.transpose(image, (2, 0, 1)) / 255.0
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    output = torch.sigmoid(output).cpu().numpy()
    output = (output > 0.5).astype(np.uint8).squeeze(0).squeeze(0)
    output = cv2.resize(output, (image.shape[2], image.shape[1]))

    return output


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

def predict_and_evaluate(data_dir, model, transform, results_file):
    results = []

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path) or class_name == '.DS_Store':
            continue

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
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
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

                if image_path == 'images/original_data/AAC/images/image_231.jpg':
                    x=2
                    x=3

                # Calculate IoU
                try:
                    iou = calculate_iou(binary_mask_resized, gt_mask)
                except ValueError as e:
                    print(f"Skipping {image_name} due to error: {e}")
                    iou = 0

                results.append([class_name, image_name, mask_name, iou])

                # Save the predicted mask
                binary_mask_resized_img = Image.fromarray(binary_mask_resized * 255)
                binary_mask_resized_img.save(predicted_mask_path)

                print(f"Processed {image_name} in {class_name} with IoU: {iou}")

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['class', 'image_name', 'mask_name', 'iou'])
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

def main():
    # Define paths
    data_dir = 'images/original_data'
    model_path = 'unet_models/models/model_epoch_32.pth'  # Change to your model path
    results_file = 'unet_models/iou_results_augmented.csv'

    # Define the transform to apply to the input image before feeding it to the model
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load the saved model
    model = Unet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict and evaluate
    predict_and_evaluate(data_dir, model, transform, results_file)

if __name__ == "__main__":
    main()
