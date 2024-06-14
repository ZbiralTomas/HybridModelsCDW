import os
import shutil
from sklearn.model_selection import train_test_split


def split_classification_data(data_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    # Ensure the directories exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Get list of images
        images = sorted([img for img in os.listdir(class_path) if img.endswith('.jpg')])

        # Define the full paths for images
        image_paths = [os.path.join(class_path, img) for img in images]

        # Split data into train, validation, and test sets
        train_imgs, temp_imgs = train_test_split(image_paths, train_size=train_ratio, random_state=random_state)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_state)

        # Create class directories in train, val, and test directories
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)

        # Copy images to respective directories
        def copy_files(file_list, destination_dir):
            for file in file_list:
                shutil.copy(file, destination_dir)

        copy_files(train_imgs, train_class_dir)
        copy_files(val_imgs, val_class_dir)
        copy_files(test_imgs, test_class_dir)

        print(f"Copied {len(train_imgs)} train images for class {class_name}.")
        print(f"Copied {len(val_imgs)} validation images for class {class_name}.")
        print(f"Copied {len(test_imgs)} test images for class {class_name}.")


# Define paths
data_dir = 'full_classification_data'
train_dir = 'images/Classification_training_data'
val_dir = 'images/Classification_validation_data'
test_dir = 'images/Classification_test_data'

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Run the split
split_classification_data(data_dir, train_dir, val_dir, test_dir, train_ratio, val_ratio, test_ratio)

print(f"Data split into {train_dir}, {val_dir}, and {test_dir} with ratios of {train_ratio}/{val_ratio}/{test_ratio}.")
