import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import csv
from resnet_model import BasicBlock, ResNet, ResNet18
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Allows use of latex text
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, all_labels, all_predictions


def plot_confusion_matrix(labels, predictions, class_names, epoch, output_dir, accuracy):
    cm = confusion_matrix(labels, predictions)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Combine counts and percentages
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_percent[i, j]
            s = f"{c}\n({p:.1f}\%)"
            annot[i, j] = s

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Accuracy: {accuracy:.2f}\%')
    plt.savefig(os.path.join(output_dir, 'figures', f'confusion_matrix_epoch_{epoch}.pdf'))
    plt.close()

    # Save confusion matrix data
    cm_data = np.array(cm)
    cm_data_file = os.path.join(output_dir, 'csv_files', f'confusion_matrix_epoch_{epoch}.csv')
    np.savetxt(cm_data_file, cm_data, delimiter=",", fmt='%d', header=",".join(class_names), comments='')


def classify_and_evaluate(model_path, test_dir, epoch, output_dir):
    # Define the transform to apply to the input image before feeding it to the model
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 4  # Change this if you have a different number of classes
    model = ResNet18(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the test dataset
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Evaluate the model
    accuracy, all_labels, all_predictions = evaluate_model(model, test_loader, device)
    print(f"Epoch {epoch}, Test Accuracy: {accuracy:.2f}%")

    # Plot and save confusion matrix
    class_names = test_dataset.classes
    plot_confusion_matrix(all_labels, all_predictions, class_names, epoch, output_dir, accuracy)

    # Save the accuracy to CSV
    accuracy_file = os.path.join(output_dir, 'test_accuracies.csv')
    if not os.path.exists(accuracy_file):
        with open(accuracy_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'test_accuracy'])

    with open(accuracy_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, accuracy])


# Example usage
if __name__ == '__main__':
    history_file = 'csv_files/resnet_history.csv'
    model_dir = 'resnet_models'
    test_dir = 'images/Classification_test_data'  # Directory containing the test data
    output_dir = 'figures/confusion_matrices'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'csv_files'), exist_ok=True)

    history = pd.read_csv(history_file)
    for i in range(90):
        if history.iloc[i]['val_accuracy'] > 92:
            model_path = os.path.join(model_dir, f'resnet_epoch_{i + 1}.pth')
            classify_and_evaluate(model_path, test_dir, i + 1, output_dir)
