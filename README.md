# CNN Model for MNIST Classification

## Final Model Architecture
We developed a CNN architecture that achieves over 95% accuracy while maintaining fewer than 25,000 parameters through several optimizations:

### Architecture Details
- **Total Parameters:** 21,614
- **Accuracy:** 95.27%
- **Target Goals:** 
  - ✓ Parameters < 25,000
  - ✓ Accuracy > 95%

### Key Components

1. **Initial Conv Block** (Strong Feature Extraction)
   - **Conv2d(1 → 8, 5×5)** with padding
   - BatchNorm2d
   - LeakyReLU(0.1)
   - **Conv2d(8 → 8, 3×3)** with padding
   - BatchNorm2d
   - LeakyReLU(0.1)
   - MaxPool2d(2)

2. **First Residual Block** (Efficient Bottleneck)
   - **Conv2d(8 → 6, 1×1)**
   - BatchNorm2d
   - LeakyReLU(0.1)
   - **Conv2d(6 → 6, 3×3)** with padding
   - BatchNorm2d
   - LeakyReLU(0.1)
   - **Conv2d(6 → 8, 1×1)**
   - BatchNorm2d
   - **Residual Connection:** Scaled by **2.5×**

3. **Transition Block**
   - LeakyReLU(0.1)
   - MaxPool2d(2)
   - **Conv2d(8 → 12, 1×1)**
   - BatchNorm2d
   - LeakyReLU(0.1)

4. **Second Residual Block** (Parameter Efficient)
   - **Conv2d(12 → 12, 3×3)** with padding, **groups=3**
   - BatchNorm2d
   - LeakyReLU(0.1)
   - **Conv2d(12 → 12, 3×3)** with padding, **groups=3**
   - BatchNorm2d
   - **Residual Connection:** Scaled by **2.0×**

5. **Classifier**
   - Flatten
   - **Linear(12×7×7 → 32)**
   - InstanceNorm1d(32)
   - LeakyReLU(0.1)
   - **Linear(32 → 10)**

### Key Design Choices
1. **Efficient Parameter Usage:**
   - **Bottleneck Blocks:** Reduce parameters while maintaining performance.
   - **Grouped Convolutions:** Limit parameter count in deeper layers.
   - **Compact Classifier:** Reduced dimensions to keep parameter count low.
   - **InstanceNorm1d:** Used in the classifier to normalize features without relying on batch statistics.

2. **Strong Feature Extraction:**
   - **Larger Initial Kernel (5×5):** Captures more spatial information in the first layer.
   - **Double Convolutions in Initial Block:** Enhances feature richness.
   - **Balanced Channel Progression:** Prevents parameter explosion while allowing deeper networks.

3. **Improved Gradient Flow:**
   - **Scaled Residual Connections:** Multiplying the identity mappings by 2.5× and 2.0× boosts gradient flow.
   - **LeakyReLU Activations:** Prevents dying ReLU problem and ensures gradients flow during training.
   - **Batch Normalization:** Applied throughout (with `track_running_stats=False`) for stable learning.

4. **Regularization:**
   - **BatchNorm Layers:** Reduce internal covariate shift.
   - **Architectural Choices:** Bottleneck structures and grouped convolutions act as implicit regularizers.
   - **No Dropout Needed:** The model achieves high accuracy without explicit dropout layers.

### Parameter Count Breakdown

- **Initial Conv Block:** 824 parameters
- **First Residual Block:** 480 parameters
- **Transition Block:** 132 parameters
- **Second Residual Block:** 936 parameters
- **Classifier:** 19,242 parameters
- **Total Parameters:** **21,614**

## Development Journey
The model evolved through several iterations focusing on balancing performance and parameter count:
1. **Baseline Model:** Started with a basic CNN exceeding 100,000 parameters.
2. **Parameter Reduction:** Introduced bottleneck blocks to decrease parameters.
3. **Residual Connections:** Added scaled residual connections to improve learning.
4. **Channel Width Optimization:** Fine-tuned channel sizes for efficiency.
5. **Grouped Convolutions:** Implemented to further reduce parameters without losing feature richness.
6. **Residual Scaling Fine-tuning:** Adjusted scaling factors to enhance gradient flow.
7. **Normalization Techniques:** Switched to InstanceNorm1d in the classifier for better generalization.

## Training Details

- **Optimizer:** Adam with learning rate `0.002`
- **Scheduler:** Cosine Annealing (`T_max=5`, `eta_min=1e-6`)
- **Loss Function:** CrossEntropyLoss
- **Training Epochs:** 1
- **Batch Size:** 64
- **Data Normalization:** Mean `0.1307`, Std `0.3081` (standard for MNIST)
- **No Data Augmentation:** Focused on model architecture improvements

## Usage

### Training

To train the model, run:

```bash
python train.py
```

The script will:

- Train the model on the MNIST dataset.
- Save the best model (based on training accuracy) in the `saved_models/` directory.

### Testing

To run the test suite, including parameter count and accuracy tests, use:

```bash
pytest tests/
```

### Model Architecture Visualization

To visualize the model architecture and parameter count:

```python
from model.network import SimpleCNN

model = SimpleCNN()
```

Upon initialization, the model will print a detailed parameter count breakdown.

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

- `torch>=2.2.0`
- `torchvision>=0.17.0`
- `pytest==7.4.3`
- `numpy>=1.24.0`

## Project Structure

- `model/`
  - `network.py`: Contains the `SimpleCNN` model definition with detailed parameter count.
  - `__init__.py`: Initializes the model module.
- `tests/`
  - `test_model.py`: Includes tests for parameter count, input/output shapes, and model accuracy.
- `train.py`: Script to train the model.
- `requirements.txt`: Lists project dependencies.
- `.github/workflows/ml-pipeline.yml`: GitHub Actions workflow for CI/CD.

## Contributing

We welcome contributions to improve the model or codebase. Please open an issue or submit a pull request with your changes.

## License

This project is licensed under the MIT License.
