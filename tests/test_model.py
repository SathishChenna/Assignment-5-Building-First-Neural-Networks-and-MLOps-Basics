import torch
from model.network import SimpleCNN, count_parameters
from torchvision import datasets, transforms
import pytest

def test_parameter_count():
    model = SimpleCNN()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_input_shape():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert True, "Model successfully processes 28x28 input"
    except:
        assert False, "Model failed to process 28x28 input"

def test_output_shape():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape[1] == 10, f"Output shape is {output.shape[1]}, should be 10"

def test_model_accuracy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    
    # Load the latest saved model
    import glob
    import os
    
    model_files = glob.glob('saved_models/model_*.pth')
    if not model_files:
        pytest.skip("No saved model found to test accuracy")
    
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Model accuracy is {accuracy:.2f}%, should be > 95%"

if __name__ == "__main__":
    # Run all tests manually
    print("Testing parameter count...")
    test_parameter_count()
    print("Parameter count test passed!")
    
    print("\nTesting input shape...")
    test_input_shape()
    print("Input shape test passed!")
    
    print("\nTesting output shape...")
    test_output_shape()
    print("Output shape test passed!")
    
    print("\nTesting model accuracy...")
    try:
        test_model_accuracy()
        print("Accuracy test passed!")
    except Exception as e:
        print(f"Accuracy test failed: {str(e)}")
    
    print("\nAll manual tests completed!")