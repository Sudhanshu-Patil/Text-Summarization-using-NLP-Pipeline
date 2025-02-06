import json
import re
from collections import defaultdict

# Load domain knowledge base
with open("domain_knowledge.json", "r") as file:
    domain_knowledge = json.load(file)

# Function for dictionary lookup
def dictionary_lookup(text, knowledge_base):
    extracted_entities = defaultdict(list)
    for category, keywords in knowledge_base.items():
        for keyword in keywords:
            if re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
                extracted_entities[category].append(keyword)
    return extracted_entities

# Rule-based NER (regex expansion)
def rule_based_ner(text, knowledge_base):
    rules = {
        "competitors": r"(Competitor[A-Z])",
        "features": r"(" + "|".join(map(re.escape, knowledge_base.get("features", []))) + ")",
        "pricing_keywords": r"(" + "|".join(map(re.escape, knowledge_base.get("pricing_keywords", []))) + ")",
    }
    extracted_entities = defaultdict(list)
    for category, pattern in rules.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        extracted_entities[category].extend(matches)
    return extracted_entities

# Combine dictionary lookup and rule-based NER
def extract_entities(text, knowledge_base):
    # Step 1: Dictionary lookup
    dict_entities = dictionary_lookup(text, knowledge_base)
    
    # Step 2: Rule-based NER
    rule_entities = rule_based_ner(text, knowledge_base)
    
    # Combine results
    final_entities = defaultdict(set)
    for category in dict_entities.keys() | rule_entities.keys():
        final_entities[category].update(dict_entities[category])
        final_entities[category].update(rule_entities[category])
    
    # Convert sets to lists for JSON serialization
    final_entities = {key: list(value) for key, value in final_entities.items()}
    return final_entities
