import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Basic transformations
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
    
    # Using SGD with Nesterov momentum
    optimizer = optim.SGD(model.parameters(), 
                         lr=0.01,
                         momentum=0.9,
                         nesterov=True,
                         weight_decay=1e-4)
    
    # Step LR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        steps_per_epoch=len(train_loader),
        epochs=1,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='linear'
    )
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
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
    
        epoch_accuracy = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f'\nEpoch {epoch} Summary:')
        print(f'Average Loss: {epoch_loss:.6f}')
        print(f'Final Accuracy: {epoch_accuracy:.2f}%\n')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'saved_models/model_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    
    return epoch_accuracy, epoch_loss

if __name__ == '__main__':
    os.makedirs('saved_models', exist_ok=True)
    final_accuracy, final_loss = train()
    print(f"Training completed!")
    print(f"Final Training Accuracy: {final_accuracy:.2f}%")
    print(f"Final Training Loss: {final_loss:.6f}") 