import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Initial conv blocks - stronger early feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),  # Stronger first layer
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),  # Extra early conv
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)  # 28x28 -> 14x14
        )
        
        # First residual block (slimmer)
        self.res1 = nn.Sequential(
            nn.Conv2d(8, 6, kernel_size=1),  # Reduce channels
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.1),
            nn.Conv2d(6, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.1),
            nn.Conv2d(6, 8, kernel_size=1),  # Back to 8
            nn.BatchNorm2d(8)
        )
        
        # Transition block
        self.trans1 = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Conv2d(8, 10, kernel_size=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.1)
        )
        
        # Second residual block (efficient)
        self.res2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3, padding=1, groups=2),  # Grouped conv
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.1),
            nn.Conv2d(10, 10, kernel_size=3, padding=1, groups=2),  # Grouped conv
            nn.BatchNorm2d(10)
        )
        
        # Classifier (efficient)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 7 * 7, 32),  # Smaller but focused
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 10)
        )
        
        # Print parameter count at initialization
        self.print_param_count()
        
    def forward(self, x):
        x = self.conv1(x)
        
        # First residual with strong connection
        identity = x
        x = self.res1(x)
        x = x + 1.5 * identity  # Stronger early residual
        
        x = self.trans1(x)
        
        # Second residual
        identity = x
        x = self.res2(x)
        x = x + identity
        
        x = self.classifier(x)
        return x
        
    def print_param_count(self):
        """Print detailed parameter count breakdown."""
        conv1_params = sum(p.numel() for p in self.conv1.parameters())
        res1_params = sum(p.numel() for p in self.res1.parameters())
        trans1_params = sum(p.numel() for p in self.trans1.parameters())
        res2_params = sum(p.numel() for p in self.res2.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        print("\nParameter count breakdown:")
        print(f"Initial conv block:  {conv1_params:,}")
        print(f"First residual:      {res1_params:,}")
        print(f"Transition:          {trans1_params:,}")
        print(f"Second residual:     {res2_params:,}")
        print(f"Classifier:          {classifier_params:,}")
        print("-" * 30)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters:    {total_params:,}\n")
        return total_params

def count_parameters(model):
    """External function to count parameters."""
    return model.print_param_count()