from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from ner_extraction import extract_entities

app = Flask(__name__)

# Load the trained model, vectorizer, and label binarizer
model = joblib.load('models/multi_label_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
mlb = joblib.load('models/multi_label_binarizer.pkl')

# Load domain knowledge base
with open("domain_knowledge.json", "r") as file:
    domain_knowledge = json.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_snippet = request.form['text_snippet']
    X = vectorizer.transform([text_snippet])
    y_pred = model.predict(X)
    labels = mlb.inverse_transform(y_pred)[0]
    
    # Extract entities
    entities = extract_entities(text_snippet, domain_knowledge)
    
    # Generate the next ID
    next_id = get_next_id()
    
    result = {
        'id': int(next_id),  # Ensure the ID is a regular integer
        'text_snippet': text_snippet,
        'labels': labels,
        'entities': entities
    }
    
    return jsonify(result)

def get_next_id():
    # File to store the last used ID
    id_file = 'data/last_id.txt'
    
    try:
        with open(id_file, 'r') as file:
            last_id = int(file.read().strip())
    except FileNotFoundError:
        last_id = 0
    
    next_id = last_id + 1
    
    with open(id_file, 'w') as file:
        file.write(str(next_id))
    
    return next_id

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
