# Pashto Handwritten Numerals OCR Dataset

## Overview
This repository contains a structured dataset for Pashto handwritten numeral recognition, prepared for machine learning and computer vision research. The dataset is provided in efficient binary formats to support fast loading and training workflows.

---

## Dataset Contents

```
data/
├── pashto_ocr.h5        # HDF5 dataset (images + labels)
├── pashto_ocr.pkl       # Pickle version of dataset
├── label_map.json       # Label-to-class mapping
```

---

## Dataset Metadata

### Number of Samples
- Total images: 10092

### Image Resolution
- Width: 32 pixels
- Height: 32 pixels
- Channels: 1 (Grayscale)

### Classes / Numerals
- Total classes: 10
- Numeral system: Pashto handwritten numerals

Example label mapping:

```json
{
  "0": "۰",
  "1": "۱",
  "2": "۲"
}
```

---

## Data Format

### HDF5 Format (.h5)
The dataset is stored with the following structure:

- `images`: NumPy array of shape `(N, H, W[, C])`
- `labels`: NumPy array of shape `(N,)`

### Pickle Format (.pkl)

Stored as a Python tuple:

(images, labels)

- `images`: NumPy array of shape `(N, H, W[, C])`
- `labels`: NumPy array of shape `(N,)`

{
    "images": numpy.ndarray,
    "labels": numpy.ndarray
}

---

## Preprocessing Steps

The dataset has undergone the following preprocessing pipeline:

1. Image collection from handwritten sources 
2. Cropping and alignment of numeral regions 
3. Resizing to fixed resolution 
4. Normalization (pixel scaling)
5. Label encoding using `label_map.json`
6. Conversion to HDF5 and Pickle formats for efficient loading

---

## Usage Example

### Load HDF5

```python
import h5py
import numpy as np 

with h5py.File("data/pashto_ocr.h5", "r") as f:
    images = np.array(f["images"])
    labels = np.array(f["labels"])
```

### Load Pickle

```python
import pickle

with open("data/pashto_ocr.pkl", "rb") as f:
    data = pickle.load(f)

	images = data[0]
	labels = data[1]
```

### Load Label Map

```python
import json

with open("data/label_map.json") as f:
    label_map = json.load(f)
```

---
### Load Pickle from GitHub

```python
import requests
import pickle
from io import BytesIO

url = "https://raw.githubusercontent.com/abasitkhan/pashto_hw_numerals/main/data/pashto_ocr.pkl"

response = requests.get(url)
images, labels = pickle.load(BytesIO(response.content))
```

### Load Label Map from GitHub

```python
import requests

url = "https://raw.githubusercontent.com/abasitkhan/pashto_hw_numerals/main/data/labels.json"

label_map = requests.get(url).json()
```

### Get the Text Label

```python
label = labels[0]
print(label)                  # Numeric class ID
print(label_map[str(label)])  # Corresponding Pashto numeral name



## Applications

- Optical Character Recognition (OCR)  
- Handwritten digit classification  
- Low-resource language processing  
- Deep learning model benchmarking  

---

## Notes

- Ensure sufficient memory when loading full dataset  
- Use batch loading for large-scale training  
- Prefer HDF5 for deep learning pipelines  

---

## License

[ADD LICENSE HERE]

---

## Citation

If you use this dataset in your research, please cite:

```
[ADD CITATION INFORMATION]
```

---

## Author

Abdul Basit

---

## Version

- v1.0 — Initial dataset release
