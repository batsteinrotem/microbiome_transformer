import pandas as pd
from sklearn.model_selection import train_test_split

# Import the custom classes we created
from TaxonomyMerger import TaxonomyMerger
from Normalize import Normalizer

# 1. Load the data (Abundances + Metadata)
print("Loading data...")
df_abundances = pd.read_csv(r"C:\Users\batst\OneDrive\المستندات\CRC_Healthy_Merged\GuptaA_2019_abundances.csv")
df_metadata = pd.read_csv(r"C:\Users\batst\OneDrive\المستندات\CRC_Healthy_Merged\GuptaA_2019_metadata.csv")

# Extract the medical label and merge it into the full dataframe
labels = df_metadata[['sample_id', 'study_condition']]
df_full = pd.merge(df_abundances, labels, on='sample_id')

# Define the target column for stratification
y = df_full['study_condition']

# 2. Split into Train, Validation, Test with Stratify
print("Splitting data...")
df_train, df_test = train_test_split(df_full, test_size=0.2, random_state=42, stratify=y)

y_train = df_train['study_condition']
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42, stratify=y_train)

# Separate labels (y) from features (X) prior to processing
y_train = df_train.pop('study_condition')
y_val = df_val.pop('study_condition')
y_test = df_test.pop('study_condition')

# 3. Phase A: Taxonomy Merge (Sub-PCA)
print("Performing taxonomy merge...")
merger = TaxonomyMerger(method='pca')
merged_train = merger.fit_transform(df_train)
merged_val = merger.transform(df_val)
merged_test = merger.transform(df_test)

# 4. Phase B: Normalization (CLR)
print("Performing CLR normalization...")
normalizer = Normalizer(method='clr')
final_train = normalizer.fit_transform(merged_train)
final_val = normalizer.transform(merged_val)
final_test = normalizer.transform(merged_test)

# 5. Export to CSV
print("Exporting files...")
# Reinsert the label (Healthy/CRC) as the second column for easier reading in Excel
final_train.insert(1, 'study_condition', y_train.values)
final_val.insert(1, 'study_condition', y_val.values)
final_test.insert(1, 'study_condition', y_test.values)

# Save the files
final_train.to_csv("Processed_Train_Data.csv", index=False)
final_val.to_csv("Processed_Validation_Data.csv", index=False)
final_test.to_csv("Processed_Test_Data.csv", index=False)

print("✅ Preprocessing completed successfully! Files saved to the directory.")