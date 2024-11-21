# CNN Model for MNIST Classification

## Final Model Architecture
We developed a CNN architecture that achieves >95% accuracy while maintaining <25,000 parameters through several optimizations:

### Architecture Details
- **Parameters:** 21,614
- **Accuracy:** 95.27%
- **Target Goals:** 
  - ✓ Parameters < 25,000
  - ✓ Accuracy > 95%

### Key Components

1. **Initial Conv Block** (Strong Feature Extraction)
   - Conv2d(1→8, 5×5) with padding
   - BatchNorm + LeakyReLU
   - Conv2d(8→8, 3×3) with padding
   - BatchNorm + LeakyReLU
   - MaxPool2d(2)

2. **First Residual Block** (Efficient Bottleneck)
   - Bottleneck: 8→6→6→8 channels
   - 1×1 convs for channel reduction/expansion
   - 3×3 conv in the middle
   - Strong residual connection (1.75×)

3. **Transition Block**
   - MaxPool2d for dimension reduction
   - Channel expansion: 8→12
   - BatchNorm + LeakyReLU

4. **Second Residual Block** (Parameter Efficient)
   - Grouped convolutions (3 groups)
   - Maintains 12 channels
   - Moderate residual connection (1.25×)

5. **Classifier**
   - Flatten
   - Linear(12×7×7 → 32)
   - BatchNorm + LeakyReLU
   - Linear(32 → 10)

### Key Design Choices
1. **Efficient Parameter Usage:**
   - Bottleneck blocks
   - Grouped convolutions
   - Compact classifier

2. **Strong Feature Extraction:**
   - Larger initial kernel (5×5)
   - Double conv in initial block
   - Balanced channel progression

3. **Gradient Flow:**
   - Scaled residual connections
   - LeakyReLU activation
   - BatchNorm throughout

4. **Regularization:**
   - BatchNorm layers
   - Architectural choices (bottleneck, groups)
   - No dropout needed

## Development Journey
The model evolved through several iterations:
1. Basic CNN (>100K parameters)
2. Parameter reduction with bottleneck blocks
3. Addition of residual connections
4. Optimization of channel widths
5. Introduction of grouped convolutions
6. Fine-tuning of residual scaling

## Usage
