 `data/` : Dataset description (data not publicly shared)
 Due to ethical and privacy constraints, the echocardiogram dataset cannot be
publicly released. Access may be granted upon reasonable request.
## Dataset Description

Dataset consists of transthoracic echocardiogram (ECHO) images collected for
the study of Rheumatic Heart Disease (RHD). The dataset contains 2D grayscale
echocardiographic frames acquired from standard clinical views.

### Data Characteristics
- Modality: Transthoracic Echocardiography
- Image type: 2D grayscale images
- Clinical focus: Rheumatic Heart Disease (RHD) analysis
- Resolution: Varies across samples (standardized during preprocessing)

### Preprocessing
All images undergo preprocessing that includes noise reduction, contrast
enhancement using Contrast Enhanced Dynamic Histogram Equalization (CEDHE),
and normalization before segmentation and classification.

### Segmentation
A U-Net-based segmentation model is used to extract regions of interest
from echocardiogram images. The segmented outputs are stored separately.

### Feature Extraction
Color co-occurrence matrix (CCM) features are extracted from segmented
regions to capture texture characteristics relevant to RHD identification.

### Dataset Availability
Due to patient privacy, ethical restrictions, and institutional data-sharing
policies, the complete echocardiogram dataset cannot be publicly released.
The repository includes only sample or placeholder files for demonstration.
Researchers may request access to the dataset from the corresponding author,
subject to institutional and ethical approval.

### Ethical Compliance
The dataset was collected in compliance with institutional ethical guidelines.
All patient identifiers were removed prior to analysis.
