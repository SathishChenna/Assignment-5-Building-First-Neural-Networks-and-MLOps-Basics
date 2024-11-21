import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                accuracy = 100 * correct / total
                avg_loss = running_loss / (batch_idx + 1)
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {avg_loss:.6f}\t'
                      f'Accuracy: {accuracy:.2f}%')
    
        # Print epoch-level metrics
        epoch_accuracy = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f'\nEpoch {epoch} Summary:')
        print(f'Average Loss: {epoch_loss:.6f}')
        print(f'Final Accuracy: {epoch_accuracy:.2f}%\n')
    
    # Save the model
    torch.save(model.state_dict(), 'saved_models/model.pth')
    
    return epoch_accuracy, epoch_loss

if __name__ == '__main__':
    # Create directory for saved models if it doesn't exist
    import os
    os.makedirs('saved_models', exist_ok=True)
    final_accuracy, final_loss = train()
    print(f"Training completed!")
    print(f"Final Training Accuracy: {final_accuracy:.2f}%")
    print(f"Final Training Loss: {final_loss:.6f}") 