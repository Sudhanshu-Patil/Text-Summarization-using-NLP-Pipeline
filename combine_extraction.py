from ner_extraction import extract_entities
import json

# Load domain knowledge base
with open("domain_knowledge.json", "r") as file:
    domain_knowledge = json.load(file)

def combined_extraction(text):
    extracted = extract_entities(text, domain_knowledge)
    return extracted


# Example usage
text = "We love the analytics, but CompetitorX has a cheaper subscription."
print(combined_extraction(text))