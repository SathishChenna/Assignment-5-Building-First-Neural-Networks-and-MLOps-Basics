import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Initial conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8, track_running_stats=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8, track_running_stats=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )
        
        # First residual block
        self.res1 = nn.Sequential(
            nn.Conv2d(8, 6, kernel_size=1),
            nn.BatchNorm2d(6, track_running_stats=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(6, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6, track_running_stats=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(6, 8, kernel_size=1),
            nn.BatchNorm2d(8, track_running_stats=False)
        )
        
        # Transition block
        self.trans1 = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 12, kernel_size=1),
            nn.BatchNorm2d(12, track_running_stats=False),
            nn.LeakyReLU(0.1)
        )
        
        # Second residual block
        self.res2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1, groups=3),
            nn.BatchNorm2d(12, track_running_stats=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(12, 12, kernel_size=3, padding=1, groups=3),
            nn.BatchNorm2d(12, track_running_stats=False)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 7 * 7, 32),
            nn.BatchNorm1d(32, track_running_stats=False),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 10)
        )
        
        # Print parameter count at initialization
        self.print_param_count()
        
    def forward(self, x):
        x = self.conv1(x)
        
        # First residual with stronger connection
        identity = x
        x = self.res1(x)
        x = x + 2.5 * identity  # Increased scaling for faster feature learning
        
        x = self.trans1(x)
        
        # Second residual with stronger connection
        identity = x
        x = self.res2(x)
        x = x + 2.0 * identity  # Increased scaling here too
        
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