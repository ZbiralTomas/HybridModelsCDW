import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import pandas as pd
from unet_model import Unet


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


def calculate_f1(predicted_mask, gt_mask):
    tp = np.sum(predicted_mask * gt_mask)
    fp = np.sum(predicted_mask * (1 - gt_mask))
    fn = np.sum((1 - predicted_mask) * gt_mask)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1


def evaluate_model(data_dir, model, transform):
    images_path = os.path.join(data_dir, 'images')
    masks_path = os.path.join(data_dir, 'masks')

    results = []
    total_predictions = len([name for name in os.listdir(images_path) if name.endswith('.jpg')])
    current_prediction = 0

    total_iou = 0
    total_f1 = 0
    num_samples = 0

    for image_name in os.listdir(images_path):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(images_path, image_name)
            mask_name = 'mask_' + image_name.split('_')[1]
            mask_path = os.path.join(masks_path, mask_name)

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

            try:
                iou = calculate_iou(binary_mask_resized, gt_mask_2d)
                f1 = calculate_f1(binary_mask_resized, gt_mask_2d)
            except ValueError as e:
                print(f"Skipping {image_name} due to error: {e}")
                iou = 0
                f1 = 0

            results.append([image_name, mask_name, iou, f1])

            total_iou += iou
            total_f1 += f1
            num_samples += 1

            current_prediction += 1
            if current_prediction % 10 == 0:
                print(f"Progress: {current_prediction}/{total_predictions} predictions done")

    avg_iou = total_iou / num_samples
    avg_f1 = total_f1 / num_samples

    return results, avg_iou, avg_f1


def main():
    # Define paths
    history_file = 'unet_models/training_history.csv'
    model_dir = 'unet_models/models'
    test_data_dir = 'images/Segmentation_test_data'
    output_dir = 'unet_models'
    os.makedirs(output_dir, exist_ok=True)

    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load history
    history = pd.read_csv(history_file)

    results = []

    for i in range(35):
        val_iou = history.loc[i, 'validation_iou']
        if val_iou > 0.89:
            epoch = i + 1
            model_path = os.path.join(model_dir, f'model_epoch_{epoch}.pth')
            model = Unet(in_channels=3, out_channels=1)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # Evaluate model
            epoch_results, avg_iou, avg_f1 = evaluate_model(test_data_dir, model, transform)
            results.append([epoch, avg_iou, avg_f1])
            print(f"Epoch {epoch}: Test IoU: {avg_iou}, Test F1: {avg_f1}")

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['epoch', 'test_iou', 'test_f1'])
    results_df.to_csv(os.path.join(output_dir, 'iou_full_data.csv'), index=False)
    print(f"Evaluation results saved to {os.path.join(output_dir, 'iou_full_data.csv')}")


if __name__ == "__main__":
    main()