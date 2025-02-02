import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(train_images, train_labels):
    train_data = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    model = ImprovedCNN(10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, loss = {running_loss/len(train_loader)}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Entrenar un modelo CNN en PyTorch")
    parser.add_argument('train_images', type=str, help='Ruta a los tensores de imágenes de entrenamiento')
    parser.add_argument('train_labels', type=str, help='Ruta a los tensores de labels de entrenamiento')
    parser.add_argument('output_models', type=str, help='Ruta donde guardar el modelo entrenado')

    args = parser.parse_args()

    train_images = torch.load(args.train_images)
    train_labels = torch.load(args.train_labels)

    model = train(train_images, train_labels)
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('models', args.output_models))

if __name__ == "__main__":
    main()
