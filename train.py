import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enhanced transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Using Adam optimizer with adjusted parameters
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.002,  # Slightly higher initial LR
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=5,
        eta_min=1e-6
    )
    
    # Training loop
    num_epochs = 1
    best_accuracy = 0
    
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
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                accuracy = 100 * correct / total
                avg_loss = running_loss / (batch_idx + 1)
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {avg_loss:.6f}\t'
                      f'Accuracy: {accuracy:.2f}%')
        
        # Epoch summary
        epoch_accuracy = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f'\nEpoch {epoch} Summary:')
        print(f'Average Loss: {epoch_loss:.6f}')
        print(f'Final Accuracy: {epoch_accuracy:.2f}%\n')
        
        # Save best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'saved_models/model_{timestamp}.pth'
            torch.save(model.state_dict(), save_path)
        
        scheduler.step()
    
    return best_accuracy, epoch_loss

if __name__ == '__main__':
    os.makedirs('saved_models', exist_ok=True)
    final_accuracy, final_loss = train()
    print(f"Training completed!")
    print(f"Best Training Accuracy: {final_accuracy:.2f}%")
    print(f"Final Training Loss: {final_loss:.6f}") 