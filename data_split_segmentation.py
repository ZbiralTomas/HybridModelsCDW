import os
import shutil
from sklearn.model_selection import train_test_split


def split_data(img_dir, mask_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
               random_state=42):
    # Ensure the directories exist
    if not os.path.exists(train_dir):
        os.makedirs(os.path.join(train_dir, 'images'))
        os.makedirs(os.path.join(train_dir, 'masks'))
    if not os.path.exists(val_dir):
        os.makedirs(os.path.join(val_dir, 'images'))
        os.makedirs(os.path.join(val_dir, 'masks'))
    if not os.path.exists(test_dir):
        os.makedirs(os.path.join(test_dir, 'images'))
        os.makedirs(os.path.join(test_dir, 'masks'))

    # Get list of images and masks
    images = sorted([img for img in os.listdir(img_dir) if img.endswith('.jpg')])
    masks = sorted([mask for mask in os.listdir(mask_dir) if mask.endswith('.jpg')])

    # Ensure the number of images and masks match
    assert len(images) == len(masks), "Number of images and masks do not match"

    # Pair images and masks
    image_paths = [os.path.join(img_dir, img) for img in images]
    mask_paths = [os.path.join(mask_dir, mask) for mask in masks]

    # Split data into train, validation, and test sets
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(image_paths, mask_paths, train_size=train_ratio, random_state=random_state)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(temp_imgs, temp_masks, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_state)

    # Copy images and masks to respective directories
    def copy_files(file_list, destination_dir):
        for file in file_list:
            shutil.copy(file, destination_dir)

    copy_files(train_imgs, os.path.join(train_dir, 'images'))
    copy_files(train_masks, os.path.join(train_dir, 'masks'))
    copy_files(val_imgs, os.path.join(val_dir, 'images'))
    copy_files(val_masks, os.path.join(val_dir, 'masks'))
    copy_files(test_imgs, os.path.join(test_dir, 'images'))
    copy_files(test_masks, os.path.join(test_dir, 'masks'))

    print(f"Copied {len(train_imgs)} train images and masks.")
    print(f"Copied {len(val_imgs)} validation images and masks.")
    print(f"Copied {len(test_imgs)} test images and masks.")

# Define paths
data_dir = 'full_segmentation_data'
img_dir = os.path.join(data_dir, 'images')
mask_dir = os.path.join(data_dir, 'masks')
train_dir = 'images/Segmentation_training_data'
val_dir = 'images/Segmentation_validation_data'
test_dir = 'images/Segmentation_test_data'

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Run the split
split_data(img_dir, mask_dir, train_dir, val_dir, test_dir)

print(f"Data split into {train_dir}, {val_dir}, and {test_dir} with ratios of {train_ratio}/{val_ratio}/{test_ratio}.")
