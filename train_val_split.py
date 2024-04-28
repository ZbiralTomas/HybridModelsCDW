import os
import os
import random
import shutil

# Define the directory containing the full data
full_data_dir = 'full_data'

# Define the directories for training and validation data
train_dir = 'training_data'
val_dir = 'validation_data'

# Create directories for training and validation data if they don't exist
for directory in [train_dir, val_dir]:
    os.makedirs(os.path.join(directory, 'images'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'masks'), exist_ok=True)

# Define the split ratio
split_ratio = 0.8

# Loop through the subdirectories containing images and masks
for root, dirs, files in os.walk(os.path.join(full_data_dir, 'images')):
    for file in files:
        # Check if the corresponding mask exists
        mask_file = os.path.join(root.replace('images', 'masks'), file.replace('image', 'mask'))
        if os.path.exists(mask_file):
            # Move to either training or validation directory based on the split ratio
            if random.random() < split_ratio:
                destination_dir = train_dir
            else:
                destination_dir = val_dir

            # Copy image and mask to the destination directory
            shutil.copy(os.path.join(root, file), os.path.join(destination_dir, 'images', file))
            shutil.copy(mask_file, os.path.join(destination_dir, 'masks', file))

print("Data split and copied successfully!")
