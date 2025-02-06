import pandas as pd
from sklearn.utils import resample

df = pd.read_csv('data/cleaned_calls_dataset.csv')

# Identify minority labels
label_counts = df['labels'].str.split(',').explode().value_counts()
minority_labels = label_counts[label_counts < 10].index.tolist()

# Augment data for minority labels
augmented_data = []
for label in minority_labels:
    subset = df[df['labels'].str.contains(label)]
    augmented_subset = resample(subset, replace=True, n_samples=10, random_state=42)
    augmented_data.append(augmented_subset)

augmented_df = pd.concat(augmented_data)
df = pd.concat([df, augmented_df])
df.to_csv('data/augmented_calls_dataset.csv', index=False)