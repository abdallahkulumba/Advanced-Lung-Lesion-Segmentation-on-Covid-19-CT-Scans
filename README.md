# Lung Lesion Segmentation from CT Scans

## Overview
This project focuses on the segmentation of lung lesions from CT scan images using deep learning techniques. The goal is to accurately identify and delineate lesions in medical images, which can assist in the diagnosis and treatment of lung-related diseases such as COVID-19.

## Dataset
The project utilizes the "COVID-19 CT Scan Lesion Segmentation Dataset" from Kaggle. The dataset contains:
- CT scan images (frames)
- Corresponding segmentation masks
- Images are in PNG format

### Data Preparation
1. **Download and Extraction**: Dataset is downloaded using Kaggle API and extracted to `CT_Dataset/`
2. **Preprocessing**:
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
   - Gaussian smoothing to reduce noise
   - Removal of completely black masks (no lesion information)
3. **Data Splitting**: 80% train, 20% test
4. **Augmentation**: Applied to small lesions (<1% of image size) including horizontal/vertical flips and rotations

## Models Implemented
The project implements and compares multiple U-Net variants with attention mechanisms:

### 1. TransUNet
- Combines Transformer encoder with U-Net decoder
- Uses ResNet50 backbone for skip connections
- Patch embedding and transformer layers for global context

### 2. Attention U-Net
- Standard U-Net with attention gates
- Squeeze-and-Excitation (SE) blocks
- Residual connections in encoder

### 3. Improved Attention U-Net
- Enhanced attention gates with multi-scale features
- Deeper architecture (5 levels)
- Advanced SE blocks and residual connections

## Training Details
- **Image Size**: 256x256 pixels
- **Batch Size**: 8
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 0.0005 with ReduceLROnPlateau scheduler
- **Loss Function**: Combined BCE + Dice Loss (90% BCE, 10% Dice)
- **Metrics**: Dice Coefficient, IoU Score
- **Early Stopping**: Patience of 5 epochs
- **Epochs**: 35-60 depending on model

## Evaluation Results
Performance on test set:

| Model | Dice Coefficient | IoU Score | Loss |
|-------|------------------|-----------|------|
| TransUNet | 0.85 | 0.78 | 0.12 |
| Attention U-Net | 0.87 | 0.80 | 0.10 |
| Improved Attention U-Net | 0.89 | 0.82 | 0.08 |

## Project Structure
```
lung-lesion-segmentation/
├── lung.ipynb                 # Main notebook with full pipeline
├── copy.ipynb                 # Additional experiments
├── latest.ipynb               # Latest version
├── CT_Dataset/                # Original dataset
│   ├── frames/               # CT scan images
│   ├── masks/                # Ground truth masks
│   ├── train/                # Training split
│   └── test/                 # Test split
├── Filtered_images/           # Preprocessed images
├── augmented_train/           # Augmented training data
├── augmented_test/            # Augmented test data
├── models/                    # Saved model weights
│   ├── model.pth             # TransUNet weights
│   ├── model1.pth            # Attention U-Net weights
│   └── model3.pth            # Improved Attention U-Net weights
└── README.md                 # This file
```

## Requirements
- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- opencv-python
- matplotlib
- scikit-learn
- albumentations
- tqdm
- kaggle (for dataset download)

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Kaggle API credentials in `kaggle.json`
4. Run the notebook cells in order

## Usage
1. **Data Preparation**: Run cells 1-4 in `lung.ipynb` to download and preprocess data
2. **Model Training**: Execute training cells for desired model
3. **Evaluation**: Run evaluation cells to test model performance
4. **Visualization**: Use visualization functions to compare predictions

## Key Features
- Comprehensive data preprocessing pipeline
- Multiple state-of-the-art segmentation models
- Data augmentation for improved generalization
- Detailed evaluation metrics and visualizations
- Early stopping and learning rate scheduling

## Future Improvements
- Implement ensemble methods
- Add more advanced augmentation techniques
- Explore 3D segmentation approaches
- Integrate with medical imaging frameworks

## References
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
- Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., ... & Zhou, Y. (2021). TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. arXiv.
- Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., ... & Rueckert, D. (2018). Attention U-Net: Learning Where to Look for the Pancreas. MIDL.

## License
This project is for educational and research purposes. Please cite appropriately if used in publications.
