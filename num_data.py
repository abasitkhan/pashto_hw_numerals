import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('pashto_alphabets_hw.csv')

# Explore the dataset structure
#print(f"Dataset shape: {df.shape}")
#print(f"Column names: {df.columns[0]}")

label_column='alif'

# Check the distribution of classes
class_counts = df[label_column].value_counts()
#print(f"Class distribution:\n{class_counts}")
#print(f"Total classes: {len(class_counts)}")


selected_classes=["yaw", "dwa", "dre", "celor", "pinza",
        "shpazh", "owa",  "ata", "nah", "las"]

#selected_classes = class_counts.head(10).index.tolist()
#print(f"Selected classes: {selected_classes}")
  # Filter dataset for selected classes
print("Filtering dataset for selected classes...")
filtered_df = df[df[label_column].isin(selected_classes)].copy()

#filtered_df = df[df.iloc[:, 0].isin(selected_classes)]
#print("Filtered shape:", filtered_df.shape)
#print("Classes after filter:", filtered_df.iloc[:, 0].unique())
 # Reset index
filtered_df.reset_index(drop=True, inplace=True)


# Create class mapping - PRESERVING THE EXACT ORDER FROM selected_classes
class_mapping = {cls: i for i, cls in enumerate(selected_classes)}
inverse_mapping = {i: cls for i, cls in enumerate(selected_classes)}

# Add numeric labels to dataframe using the preserved order
filtered_df['class_id'] = filtered_df[label_column].map(class_mapping)
#print(f"Filtered dataset shape: {filtered_df.shape}")
#print(f"Classes distribution in filtered data:")
#print(filtered_df[label_column].value_counts())
print("Splitting into train and test sets...")
#train_df, test_df = train_test_split(
#    filtered_df, 
#    test_size=0.25, 
#    random_state=42,
#    stratify=filtered_df[label_column]
#)
#
#print(f"Training set: {train_df.shape[0]} samples")
#print(f"Test set: {test_df.shape[0]} samples")

# Save datasets based on selected format
base_name = "10class_dataset"

output_format = 'csv'

## Saving all 10 classess in CSV format
if output_format in ['csv', 'all']:
    print("\nSaving as CSV files...")
    # Save the complete dataset
    filtered_df.to_csv(f'{base_name}_complete.csv', index=False)
    
    # Save class mapping with preserved order
    mapping_df = pd.DataFrame([
        {'class_id': i, 'class_name': cls} 
        for i, cls in enumerate(selected_classes)
    ])
    mapping_df.to_csv(f'{base_name}_mapping.csv', index=False)
    
    print(f"Saved: {base_name}_complete.csv")
    print(f"Saved: {base_name}_mapping.csv")


# Load data
df = pd.read_csv("10class_dataset_complete.csv")
mapping_df = pd.read_csv("10class_dataset_mapping.csv")

print(f"Dataset shape: {df.shape}")
print("Class mapping:")
#print(mapping_df)
print("Classes after filter:", df.iloc[:, 0].unique())

## FINAL VERIFICATION: Compare expected vs actual mapping
#print("\n" + "="*50)
#print("FINAL VERIFICATION: Expected vs Actual Class Mapping")
#print("="*50)
#print("Expected order (from your list):")
#for i, class_name in enumerate(selected_classes):
#    print(f"  Class ID {i}: '{class_name}'")
#
#print("\nActual mapping in dataset:")
#actual_mapping = {i: cls for i, cls in enumerate(selected_classes)}
#for class_id, class_name in sorted(actual_mapping.items()):
#    print(f"  Class ID {class_id}: '{class_name}'")




#print("\nDataset extraction completed successfully!")
#print(f"Selected classes (in order): {selected_classes}")
#print(f"Class mapping (preserved order): {class_mapping}")
#print(f"Total samples: {len(filtered_df)}")
#print(f"Training samples: {len(train_df)}")
#print(f"Test samples: {len(test_df)}")


