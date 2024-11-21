# MNIST CNN with MLOps Pipeline

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline. The model achieves >95% accuracy in a single epoch while maintaining less than 25,000 parameters.

## Project Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest

## Model Architecture

The model is a simple CNN with:
- 2 convolutional layers with batch normalization
- 2 fully connected layers
- Less than 25,000 trainable parameters
- Designed for 28x28 grayscale input images
- 10-class output (digits 0-9)

## Project Structure
project/
├── model/
│ ├── init.py
│ └── network.py
├── tests/
│ └── test_model.py
├── saved_models/
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml
├── train.py
├── requirements.txt
└── .gitignore

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```



## Usage

1. Train the model:
```bash
python train.py
```

2. Run tests:
```bash
python -m pytest tests/
```



## Model Specifications

- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Parameters: < 25,000
- Training Accuracy: > 95%
- Training Epochs: 1

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Runs tests to verify:
   - Parameter count (< 25,000)
   - Input shape handling (28x28)
   - Output shape (10 classes)
   - Model accuracy (> 95%)
5. Archives the trained model

## Test Coverage

Tests verify:
- Model parameter count is under 25,000
- Model correctly handles 28x28 input images
- Model outputs 10 classes
- Model achieves >95% accuracy on training data

## Model Saving

Trained models are saved with timestamps in the `saved_models` directory:
