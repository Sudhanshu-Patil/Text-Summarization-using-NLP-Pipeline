import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Load the dataset
df = pd.read_csv('calls_dataset.csv')

# Apply preprocessing to the text snippets
df['cleaned_text'] = df['text_snippet'].apply(preprocess_text)

# Create a new folder to save the output CSV files
output_folder = 'data'
os.makedirs(output_folder, exist_ok=True)

# Save the cleaned dataset in the new folder
output_file_path = os.path.join(output_folder, 'cleaned_calls_dataset.csv')
df.to_csv(output_file_path, index=False)

print(f"Cleaned dataset saved to {output_file_path}")