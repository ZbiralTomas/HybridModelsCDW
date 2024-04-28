from rembg import remove
import cv2
import os
import numpy as np
from PIL import Image
import re

def sort_by_number(path):
    # Extract the numeric part of the filename using regular expressions
    match = re.search(r'image_(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        return float('inf')  # Return infinity if no numeric part found



data_dir = 'data'
all_contents_of_data_dir = os.listdir(data_dir)
class_list = [item for item in all_contents_of_data_dir if os.path.isdir(os.path.join(data_dir, item))]
class_list.sort()

for class_name in class_list:
    class_dir = os.path.join(data_dir, class_name)

    image_subdir = os.path.join(class_dir, 'images')
    save_mask_subdir = os.path.join(class_dir, 'masks')
    os.makedirs(save_mask_subdir, exist_ok=True)
    image_list = os.listdir(image_subdir)
    image_list = sorted(image_list, key=sort_by_number)

    for index, image in enumerate(image_list, 1):
        if image.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_subdir, image)
            mask_path = save_mask_subdir+'/mask_'+str(index)+'.jpg'
            img = cv2.imread(image_path)
            img_removed = remove(img)
            result_array = np.array(img_removed)
            alpha_channel = result_array[:, :, 3]
            mask = (alpha_channel > 0).astype(np.uint8) * 255
            mask_image = Image.fromarray(mask)
            image_np = np.array(mask_image)
            cv2.imwrite(mask_path, image_np)
            print(class_name+' '+str(index)+'/'+str(len(image_list)))

