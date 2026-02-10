# CEDHE-GAN-RHD1
about RHD
# CEDHE-GAN for Rheumatic Heart Disease Identification

This repository provides the implementation of the proposed framework for
echocardiogram image enhancement, segmentation, feature extraction, and
classification for Rheumatic Heart Disease (RHD) identification.

## 📌 Components
- Image preprocessing and enhancement (CEDHE-GAN)
- U-Net based segmentation
- Color co-occurrence matrix (CCM) feature extraction
- CNN-based classification

## 📂 Repository Structure
- `data/` : Dataset description (data not publicly shared)
- `preprocessing/` : Image enhancement and preprocessing scripts
- `segmentation/` : U-Net model, training metrics, and segmented outputs
- `features/` : Extracted color co-occurrence features
- `classification/` : CNN classification model and scripts

## 📊 Dataset Availability
Due to ethical and privacy constraints, the echocardiogram dataset cannot be
publicly released. Access may be granted upon reasonable request.

## ⚙️ Requirements
```bash
pip install -r requirements.txt
