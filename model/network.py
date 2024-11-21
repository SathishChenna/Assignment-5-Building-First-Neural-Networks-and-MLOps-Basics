import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        
        # Fully connected layers
        self.fc1 = nn.Linear(20 * 7 * 7, 50)
        self.fc2 = nn.Linear(50, 10)
        
        # Regularization
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.3)
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout2(x)
        
        # Fully connected layers
        x = x.view(-1, 20 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

model = SimpleCNN()
print(f"Total parameters: {count_parameters(model)}") 