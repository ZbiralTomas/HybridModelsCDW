import os
from skimage import io
from skimage.transform import rotate
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import cv2


def augment_image(image, mask):
    # Define the augmentation pipeline for both image and mask
    transform = A.Compose([
        A.Rotate(limit=25, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.OneOf([
            A.Blur(blur_limit=5, p=0.2),
            A.GaussianBlur(blur_limit=5, p=0.2),
        ], p=0.2),
        ToTensorV2(),
    ])

    # Apply augmentation to both image and mask
    augmented = transform(image=image, mask=mask)

    # Retrieve the augmented image and mask
    augmented_image = augmented['image']
    augmented_mask = augmented['mask']

    # Convert the augmented image and mask to NumPy arrays for display
    augmented_image_np = augmented_image.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
    augmented_mask_np = augmented_mask.squeeze().numpy()  # Remove the channel dimension for mask

    return augmented_image_np, augmented_mask_np


data_dir = 'data'
save_dir = 'augmented_data'
all_contents_of_data_dir = os.listdir(data_dir)
class_list = [item for item in all_contents_of_data_dir if os.path.isdir(os.path.join(data_dir, item))]
class_list.sort()

for class_name in class_list:
    class_dir = os.path.join(data_dir, class_name)
    save_class_dir = os.path.join(save_dir, class_name)
    os.makedirs(save_class_dir, exist_ok=True)

    image_subdir = os.path.join(class_dir, 'images')
    mask_subdir = os.path.join(class_dir, 'masks')
    save_image_subdir = os.path.join(save_class_dir, 'images')
    os.makedirs(save_image_subdir, exist_ok=True)
    save_mask_subdir = os.path.join(save_class_dir, 'masks')
    os.makedirs(save_mask_subdir, exist_ok=True)
    image_list = os.listdir(image_subdir)
    image_list.sort()
    mask_list = os.listdir(mask_subdir)
    mask_list.sort()
    index = 0

    for image_path, mask_path in zip(image_list, mask_list):
        if image_path.endswith(('.jpg', '.jpeg', '.png')):
            index += 1
            image_path = os.path.join(image_subdir, image_path)
            mask_path = os.path.join(mask_subdir, mask_path)
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)
            augmented_image, augmented_mask = augment_image(image, mask)
            save_path_aug_image = save_image_subdir+f'/image_{index}.jpg'
            save_path_aug_mask = save_mask_subdir+f'/mask_{index}.jpg'
            cv2.imwrite(save_path_aug_image, augmented_image)
            cv2.imwrite(save_path_aug_mask, augmented_mask)
            print(class_name + ' ' + str(index) + '/' + str(len(image_list)))




