import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 14 * 14, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 14 * 14)
        x = self.fc(x)
        return x

def test(model, test_images, test_labels):
    test_data = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model.eval()
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

    report = classification_report(all_labels, all_predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv('metrics.csv', index=True)

    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Testar un modelo CNN entrenado en PyTorch")
    parser.add_argument('test_images', type=str, help='Ruta a los tensores de imágenes de prueba')
    parser.add_argument('test_labels', type=str, help='Ruta a los tensores de labels de prueba')
    parser.add_argument('model_path', type=str, help='Ruta al modelo entrenado')

    args = parser.parse_args()

    test_images = torch.load(args.test_images)
    test_labels = torch.load(args.test_labels)

    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(args.model_path))

    test(model, test_images, test_labels)

if __name__ == "__main__":
    main()
