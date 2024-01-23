import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
def load_images(image_paths):
    images = []
    for image_path in image_paths:
        full_path = r'Dataset\PCOSGen-train\PCOSGen-train\images\\' + image_path  # Use raw string (r'') to handle backslashes
        img = cv2.imread(full_path)
        if img is not None:
            img_resized = cv2.resize(img, (28, 28))
            img_tensor = torch.from_numpy(img_resized).float()
            images.append(img_tensor)
        else:
            print(f"Failed to read or resize image: {image_path}")
    return images

# Assuming you have X_train and y_train as NumPy arrays
# Modify the data loading part based on your actual data format
X_train_paths = np.load('Dataset/X_train.npy', allow_pickle=True)
y_train = np.load('Dataset/y_train.npy', allow_pickle=True)
X_test_paths = np.load('Dataset/X_test.npy', allow_pickle=True)
y_test = np.load('Dataset/y_test.npy', allow_pickle=True)
X_train = load_images(X_train_paths)
X_test = load_images(X_test_paths)
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert NumPy arrays to PyTorch tensors
X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train).long()
X_val, y_val = torch.Tensor(X_val), torch.Tensor(y_val).long()

# Define the DenseNet model
class DenseNetModel(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetModel, self).__init__()
        self.densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)

# Initialize the model
num_classes = 2  # assuming binary classification (healthy/unhealthy)
model = DenseNetModel(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
num_epochs = 10
batch_size = 32

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.to(device))
        _, predicted = torch.max(val_outputs, 1)

        val_accuracy = accuracy_score(y_val.numpy(), predicted.cpu().numpy())
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy}')

# Save the trained model
torch.save(model.state_dict(), 'densenet_model.pth')
