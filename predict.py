import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import Unet

# Define the transform to apply to the input image before feeding it to the model
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to match the input size expected by the model
    transforms.ToTensor()  # Convert the image to a PyTorch tensor
])

# Load the saved model
model = Unet()  # Initialize your model
model.load_state_dict(torch.load('saved_models/model_epoch_3.pth'))
model.eval()  # Set the model to evaluation mode

# Load and preprocess the input image
input_image = Image.open('validation_data/images/image_101.jpg')
input_tensor = transform(input_image).unsqueeze(0)  # Add a batch dimension

# Make a prediction
with torch.no_grad():
    output_mask = model(input_tensor)

# Convert the output mask to a numpy array and squeeze the batch dimension
output_mask = output_mask.squeeze(0).numpy()

# If the output mask is grayscale, reshape it to a 2D array
if output_mask.ndim == 3 and output_mask.shape[0] == 1:
    output_mask = output_mask.squeeze(0)

threshold = 0

# Threshold the output mask
binary_mask = (output_mask > threshold).astype(np.uint8)

# Visualize the input image and the binary mask
plt.figure(figsize=(10, 5))

# Plot the input image
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title('Input Image')
plt.axis('off')

# Plot the binary mask
plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary Mask')
plt.axis('off')

plt.show()