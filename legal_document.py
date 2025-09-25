# Legal Document Extractor & Summarizer - single file version

# Prerequisites:
# pip install spacy transformers

import spacy
from transformers import pipeline
import re

# Load SpaCy model (NER, sentence detection)
nlp = spacy.load("en_core_web_sm")

# Load Hugging Face summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example legal document (replace this string with your document input logic)
legal_text = """
This Agreement is entered into on this 1st day of January, 2023, by and between Alpha Corp and Beta Ltd...
Whereas Alpha Corp agrees to supply products under clause 4.1, and Beta Ltd. shall make payment as defined in clause 6.2...
Governing law shall be the laws of the Republic of India.
Confidentiality requirements are as specified in clause 8...
Limitation of liability applies as described in clause 10...
"""

# ----------- Entity/Clause Extraction -----------

print("=== Key Entities and Clauses ===")
doc = nlp(legal_text)

# Extract and display named entities
for ent in doc.ents:
    print(f"{ent.text} [{ent.label_}]")

# Regex-based clause reference extraction
clauses = re.findall(r'(clause\s+\d+\.\d+)', legal_text, re.IGNORECASE)
print("Clause References:", clauses)

# ----------- Document Summarization -----------

print("\n=== Document Summary ===")
summary = summarizer(legal_text, max_length=120, min_length=40, do_sample=False)
print(summary['summary_text'])

# ----------- Sentence Listing (Optional) -----------

print("\n=== Sentences Detected ===")
for sent in doc.sents:
    print(sent.text.strip())


