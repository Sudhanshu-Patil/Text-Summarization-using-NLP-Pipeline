import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import joblib

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_df['cleaned_text'])
X_test = vectorizer.transform(test_df['cleaned_text'])

# Binarize labels for multi-label classification
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df['labels'].str.split(','))
y_test = mlb.transform(test_df['labels'].str.split(','))


# Train model with hyperparameter tuning
param_grid = {'estimator__C': [0.1, 1, 10]}
model = OneVsRestClassifier(LogisticRegression())
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_micro')
grid_search.fit(X_train, y_train)

# Create a new folder to save the models
models_folder = 'models'
os.makedirs(models_folder, exist_ok=True)

# Save the trained model, vectorizer, and binarizer
joblib.dump(grid_search, os.path.join(models_folder, 'multi_label_model.pkl'))
joblib.dump(vectorizer, os.path.join(models_folder, 'tfidf_vectorizer.pkl'))
joblib.dump(mlb, os.path.join(models_folder, 'multi_label_binarizer.pkl'))

# Evaluate model
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))