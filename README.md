# AI-Lung-CT-Analysis
A powerful deep learning model for automated lung CT scan analysis using Convolutional Neural Networks. This model excels at detecting nodules, identifying pneumonia patterns, and analyzing COVID-19 related abnormalities.

# Lung CT Analysis CNN
## Features

- Automated detection of lung abnormalities
- Multi-class pathology classification
- Real-time visualization of detection results
- Hounsfield Unit normalization for CT scans
- Specialized CNN architecture for lung imaging
- Support for DICOM (.dcm) and NIFTI (.nii) formats

## Clinical Applications

- Lung nodule detection
- COVID-19 pattern recognition
- Pneumonia identification
- General lung structure analysis
- Early disease detection
- Clinical research support

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-image
- scikit-learn
- pydicom (for DICOM files)
- nibabel (for NIFTI files)

## Installation

```bash
git clone https://github.com/yourusername/lung-ct-analysis.git
cd lung-ct-analysis
pip install -r requirements.txt


LungCT/
├── training/
│   ├── scans/
│   │   ├── patient1.dcm
│   │   └── patient2.nii
│   └── masks/
│       ├── mask1.dcm
│       └── mask2.nii
└── ct_results/
```

mkdir -p LungCT/training/scans
mkdir -p LungCT/training/masks
mkdir -p ct_results

Run the script:
python lung_ct_cnn.py


Model Architecture

Input Layer: 512x512x1 (high-resolution CT slices)

Specialized convolutional layers for lung features

Dropout layers for robust learning

Hounsfield Unit normalization

Output Layer: Abnormality detection mask

Results Output

Abnormality detection masks

Visualization plots showing:

Original CT slice (bone colormap)
Ground truth annotations

Predicted abnormalities

Training metrics and progress

Performance Metrics

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Real-time visualization every 2 epochs

Abnormality detection accuracy

Contributing
Fork the repository

Create your feature branch (git checkout -b feature/NewFeature)
Commit your changes (git commit -m 'Add NewFeature')
Push to the branch (git push origin feature/NewFeature)
Open a Pull Request
License
MIT License - see LICENSE.md

Please talk to me before you use this in your Research.

