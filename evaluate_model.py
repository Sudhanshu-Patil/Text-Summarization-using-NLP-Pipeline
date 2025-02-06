import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, precision_score, recall_score, f1_score
import joblib
import numpy as np
import sys

# Load data
print("Loading test data...")
test_df = pd.read_csv('data/test.csv')

# Load the trained model and vectorizer
print("Loading model and vectorizer...")
grid_search = joblib.load('models/multi_label_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
mlb = joblib.load('models/multi_label_binarizer.pkl')

# Vectorize text data
print("Vectorizing text data...")
X_test = vectorizer.transform(test_df['cleaned_text'])

# Binarize labels
print("Binarizing labels...")
y_test = mlb.transform(test_df['labels'].str.split(','))

# Predict using the best model from grid_search
print("Predicting labels...")
y_pred = grid_search.predict(X_test)

# Calculate precision, recall, and F1 score with zero_division parameter
print("Calculating precision, recall, and F1 score...")
precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Generate classification report
print("Generating classification report...")
report = classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=1, output_dict=True)

# Save classification report to a CSV file
print("Saving classification report to CSV...")
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('data/classification_report.csv', index=True)

# Generate confusion matrix
print("Generating confusion matrix...")
conf_matrix = multilabel_confusion_matrix(y_test, y_pred)

# Plot heatmap of label co-occurrences
print("Plotting heatmap...")
label_co_occurrence = pd.DataFrame(y_test.T @ y_test, index=mlb.classes_, columns=mlb.classes_)
sns.heatmap(label_co_occurrence, annot=True, cmap='Blues')
plt.title('Label Co-occurrence Heatmap')
plt.savefig('data/label_co_occurrence_heatmap.png')  # Save the heatmap to a file
plt.close()

# Error analysis: Identify misclassifications
print("Identifying misclassifications...")
misclassified_indices = [i for i in range(len(y_test)) if not (y_test[i] == y_pred[i]).all()]
misclassified_samples = test_df.iloc[misclassified_indices]

# Save misclassified samples to a CSV file
print("Saving misclassified samples to CSV...")
misclassified_samples.to_csv('data/misclassified_samples.csv', index=False)

# Print examples of misclassifications
print("Examples of Misclassifications:")
for index, row in misclassified_samples.iterrows():
    print(f"Text: {row['text_snippet']}")
    print(f"True Labels: {row['labels']}")
    predicted_labels = mlb.inverse_transform(y_pred[index].reshape(1, -1))[0]
    print(f"Predicted Labels: {', '.join(predicted_labels)}")
    print("----")
