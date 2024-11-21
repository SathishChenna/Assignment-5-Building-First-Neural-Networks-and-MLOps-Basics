import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional block with fewer filters
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Additional pooling to reduce feature map size
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers with reduced dimensions
        # After 3 max pooling operations: 28x28 -> 14x14 -> 7x7 -> 3x3
        self.fc1 = nn.Linear(16 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        # Additional pooling to reduce feature map size
        x = self.pool(x)
        
        # Fully connected layers
        x = x.view(-1, 16 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

model = SimpleCNN()
print(f"Total parameters: {count_parameters(model)}") 