import pandas as pd
from combine_extraction import combined_extraction

# Load data
df = pd.read_csv('data/augmented_calls_dataset.csv')

# Extract entities for each snippet
df['extracted_entities'] = df['cleaned_text'].apply(combined_extraction)

# Save the results
df.to_csv('data/extracted_entities.csv', index=False)
