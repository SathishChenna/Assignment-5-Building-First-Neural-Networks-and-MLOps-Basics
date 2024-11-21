import torch
from model.network import SimpleCNN

def test_cnn_output_shape():
    model = SimpleCNN()
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 28, 28)  # MNIST image size
    output = model(input_tensor)
    assert output.shape == (batch_size, 10), f"Expected output shape (1, 10), got {output.shape}"

def test_model_parameters():
    model = SimpleCNN()
    assert any(param.requires_grad for param in model.parameters()), "Model has no trainable parameters" 