import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from unet_model import Unet
import csv
import time
import matplotlib.pyplot as plt


def calculate_iou(outputs, masks):
    intersection = torch.sum(outputs * masks)
    union = torch.sum((outputs + masks) > 0)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou


def calculate_f1(outputs, masks):
    tp = torch.sum(outputs * masks)
    fp = torch.sum(outputs * (1 - masks))
    fn = torch.sum((1 - outputs) * masks)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1


# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, img_folder, mask_folder, transform=None):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform

        self.img_filenames = sorted(os.listdir(img_folder))
        self.mask_filenames = sorted(os.listdir(mask_folder))

        # Ensure the number of images and masks match
        assert len(self.img_filenames) == len(self.mask_filenames), "Number of images and masks do not match"

        # Ensure each image has a corresponding mask
        for img, mask in zip(self.img_filenames, self.mask_filenames):
            img_number = os.path.splitext(img)[0].split('_')[1]
            mask_number = os.path.splitext(mask)[0].split('_')[1]
            assert img_number == mask_number, f"Image and mask {img} and {mask} are not paired"

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.img_filenames[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Apply thresholding to obtain binary mask
        mask = (mask > 0.5).float()  # Adjust threshold as needed

        return image, mask


def visualize_samples(dataset, num_samples=5):
    for i in range(num_samples):
        image, mask = dataset[i]
        image = image.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
        mask = mask.squeeze(0)  # Remove channel dimension

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[0].set_title('Image')
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title('Mask')
        plt.show()


# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Define data paths
train_img_folder = 'Segmentation_training_data/images'
train_mask_folder = 'Segmentation_training_data/masks'
val_img_folder = 'Segmentation_validation_data/images'
val_mask_folder = 'Segmentation_validation_data/masks'

# Create datasets
train_dataset = CustomDataset(train_img_folder, train_mask_folder, transform=transform)
val_dataset = CustomDataset(val_img_folder, val_mask_folder, transform=transform)

visualize_samples(train_dataset)
# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define the model, loss function, and optimizer
model = Unet(in_channels=3, out_channels=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Define training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, save_dir='unet_models',
                history_file='training_history_new.csv'):
    model_save_dir = os.path.join(save_dir, 'models')
    optimizer_save_dir = os.path.join(save_dir, 'optimizers')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(optimizer_save_dir, exist_ok=True)

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [], 'train_f1': [],
               'val_f1': [], 'training_time': []}

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        running_train_loss = 0.0
        running_train_iou = 0.0
        running_train_f1 = 0.0
        for batch_idx, (inputs, masks) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)

            # Apply sigmoid activation and thresholding to obtain binary masks
            outputs = torch.sigmoid(outputs)
            binary_outputs = (outputs > 0.5).float()

            iou = calculate_iou(binary_outputs, masks)
            f1 = calculate_f1(binary_outputs, masks)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            running_train_iou += iou.item() * inputs.size(0)
            running_train_f1 += f1.item() * inputs.size(0)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Train Loss: {loss.item()},"
                f" Train IoU: {iou.item()}, Train F1: {f1.item()}")

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_iou = running_train_iou / len(train_loader.dataset)
        epoch_train_f1 = running_train_f1 / len(train_loader.dataset)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss}, Train IoU: {epoch_train_iou},"
            f" Train F1: {epoch_train_f1}")

        # Validate the model
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_f1 = 0.0
        with torch.no_grad():
            for inputs, masks in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, masks)

                # Apply sigmoid activation and thresholding to obtain binary masks
                outputs = torch.sigmoid(outputs)
                binary_outputs = (outputs > 0.5).float()

                iou = calculate_iou(binary_outputs, masks)
                f1 = calculate_f1(binary_outputs, masks)

                val_loss += loss.item() * inputs.size(0)
                val_iou += iou.item() * inputs.size(0)
                val_f1 += f1.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_iou = val_iou / len(val_loader.dataset)
        epoch_val_f1 = val_f1 / len(val_loader.dataset)
        print(f"Validation Loss: {epoch_val_loss}, Validation IoU: {epoch_val_iou}, Validation F1: {epoch_val_f1}")

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch + 1} Training Time: {epoch_time} seconds")
        print()

        # Save history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(epoch_train_iou)
        history['val_iou'].append(epoch_val_iou)
        history['train_f1'].append(epoch_train_f1)
        history['val_f1'].append(epoch_val_f1)
        history['training_time'].append(epoch_time)

        # Save the model and optimizer state after every epoch
        model_path = os.path.join(model_save_dir, f'model_epoch_{epoch + 1}.pth')
        optimizer_path = os.path.join(optimizer_save_dir, f'optimizer_epoch_{epoch + 1}.pth')
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict()}, model_path)
        torch.save({'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict()}, optimizer_path)
        print(f"Model saved at {model_path}")
        print(f"Optimizer saved at {optimizer_path}")

    # Save history to CSV file
    with open(os.path.join(save_dir, history_file), 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_iou', 'val_iou', 'train_f1', 'val_f1', 'training_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(history['epoch'])):
            writer.writerow({fieldnames[0]: history['epoch'][i],
                             fieldnames[1]: history['train_loss'][i],
                             fieldnames[2]: history['val_loss'][i],
                             fieldnames[3]: history['train_iou'][i],
                             fieldnames[4]: history['val_iou'][i],
                             fieldnames[5]: history['train_f1'][i],
                             fieldnames[6]: history['val_f1'][i],
                             fieldnames[7]: history['training_time'][i]})
    print("Training history saved to CSV.")


# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=40, save_dir='unet_models',
            history_file='unet_history.csv')
