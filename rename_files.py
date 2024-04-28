import os

directory = 'augmented_data/Concrete/images'  # Replace 'your_directory_path' with the path to your directory

# List files in the directory
files = os.listdir(directory)

# Filter only .jpg files
jpg_files = [file for file in files if file.endswith('.jpg')]

# Sort the .jpg files
jpg_files.sort()

for i, file in enumerate(jpg_files, 1):
    old_path = os.path.join(directory, file)
    new_path = os.path.join(directory, f'image_{5093+i}.jpg')
    os.rename(old_path, new_path)
    print(f'Renamed {old_path} to {new_path}')