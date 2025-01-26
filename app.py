from flask import Flask, request, jsonify
from combine_extraction import combined_extraction
import json

app = Flask(__name__)

# Load domain knowledge base
with open("domain_knowledge.json", "r") as file:
    domain_knowledge = json.load(file)

@app.route('/extract', methods=['POST'])
def extract():
    data = request.json
    text = data.get('snippet', '')
    
    # Extract entities
    extracted_entities = combined_extraction(text)
    
    # Generate summary (simple example, can be improved)
    summary = f"Extracted {len(extracted_entities['competitors'])} competitors, {len(extracted_entities['features'])} features, and {len(extracted_entities['pricing_keywords'])} pricing keywords."
    
    response = {
        "predicted_labels": [],  # Placeholder for multi-label classification
        "extracted_entities": extracted_entities,
        "summary": summary
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)