# 10-Class Dataset Subset

## Description
This dataset contains 10 classes extracted from the original 54-class dataset. The complete dataset is provided without pre-splitting to allow flexible usage.

## Files
- `pashto_hw_numerals.csv` - Complete dataset with all samples
- `pashto_hw_numerals_mapping.csv` - Mapping between class IDs and class names
- `pashto_hw_numerals.pkl` - Pickle file containing data and metadata
- `pashto_hw_numerals.h5` - HDF5 file containing data and metadata

## Classes Included (in order):
0. yaw
1. dwa
2. dre
3. celor
4. pinza
5. shpazh
6. owa
7. ata
8. nah
9. las

## Usage Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/yourusername/your-repo/main/10class_dataset_complete.csv')

# Split as needed
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['class_id'])

## The first column contains the id of ths data
## The second column contains the text labels of the data
labels = df.iloc[:, 0].values  # First column = labels
pixels = df.iloc[:, 2:].values  # Remaining columns = pixels


