import cv2
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import csv
from resnet_model import ResNet18


def classify_images(model, dataloader, device, class_names, output_file):
    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            for idx in range(inputs.size(0)):
                image_path = dataloader.dataset.samples[batch_idx * dataloader.batch_size + idx][0]
                probability_list = probabilities[idx].tolist()
                predicted_label = predicted_labels[idx]
                true_label = labels[idx]
                is_correct = predicted_label == true_label
                result = [image_path] + probability_list + [class_names[predicted_label], is_correct]
                results.append(result)

    # Save predictions to CSV
    headers = ['image_path'] + class_names + ['predicted_label', 'is_correct']
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)

    print(f"Predictions saved to {output_file}")


def main():
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
    model_path = 'resnet_models/models/resnet_epoch_61.pth'  # Change to your model path
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the dataset
    test_dir = 'images/Classification_test_data'  # Directory containing the test data
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Class names
    class_names = test_dataset.classes

    # Output CSV file
    output_file = 'csv_files/resnet_predictions.csv'

    # Classify images and save predictions
    classify_images(model, test_loader, device, class_names, output_file)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 4  # Change this if you have a different number of classes
model = ResNet18(num_classes).to(device)
model_path = 'resnet_models/models/resnet_epoch_61.pth'  # Change to your model path
model.load_state_dict(torch.load(model_path))
model.eval()
img = cv2.imread('images/Classification_test_data/AAC/cropped_image_231.jpg')
transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

if __name__ == '__main__':
    main()