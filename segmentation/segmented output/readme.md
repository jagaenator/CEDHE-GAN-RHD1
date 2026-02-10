## Segmented Output

This folder contains the segmented echocardiogram images generated using the
trained U-Net segmentation model (`model_unet_50ep.keras`).

### Description
The segmentation process isolates clinically relevant cardiac regions from
echocardiogram images to support subsequent feature extraction and
classification tasks for Rheumatic Heart Disease (RHD) identification.

### Output Characteristics
- Segmentation model: U-Net
- Input: Preprocessed echocardiogram images
- Output: Binary or multi-class segmentation masks
- Format: Image files (e.g., PNG/JPEG)

### Usage
The segmented outputs are used as inputs for:
- Color co-occurrence matrix (CCM) feature extraction
- CNN-based classification

### Notes
Due to patient privacy and ethical constraints, only representative or
placeholder segmented images are included in this repository.
The complete set of segmented outputs is available upon reasonable request,
subject to institutional approval.

### Reproducibility
The segmentation results can be reproduced by running the segmentation scripts
using the provided U-Net model and preprocessing pipeline.
