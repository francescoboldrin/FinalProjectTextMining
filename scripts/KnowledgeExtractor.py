"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-13 11:13:30
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-13 16:45:09
FilePath: scripts/KnowledgeExtractor.py
Description: functions to extract entities and linking between entitites to create a knowledge graph
"""
DEBUG = 1

import json
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEBUG:
    print(device)


if DEBUG:
    print("pipeline built")

debug_sentences = [
    ["Elon", "Musk", "is", "the", "CEO", "of", "Tesla", "and", "SpaceX"],
    ["Jeff", "Bezos", "founded", "Amazon", "and", "he", "owns", "The", "Washington", "Post"],
    ["Microsoft", "released", "Windows", "10", "under", "the", "leadership", "of", "Satya", "Nadella"],
    ["Tim", "Cook", "introduced", "the", "iPhone", "12", "during", "an", "Apple", "event"],
    ["Mark", "Zuckerberg", "announced", "the", "rebranding", "of", "Facebook", "to", "Meta"],
    ["Bill", "Gates", "and", "Paul", "Allen", "founded", "Microsoft", "in", "1975"],
    ["Steve", "Jobs", "presented", "the", "first", "iPad", "in", "2010"],
    ["Sundar", "Pichai", "is", "the", "CEO", "of", "Google", "and", "Alphabet"],
    ["Sheryl", "Sandberg", "served", "as", "COO", "at", "Facebook", "and", "wrote", "Lean", "In"],
    ["The", "Coca-Cola", "Company", "produces", "Coke", "and", "Sprite"],
    ["Nike", "introduced", "the", "Air", "Jordan", "series", "in", "collaboration", "with", "Michael", "Jordan"],
    ["Henry", "Ford", "founded", "Ford", "Motor", "Company", "and", "revolutionized", "manufacturing"],
    ["Jack", "Ma", "started", "Alibaba", "and", "owns", "a", "stake", "in", "Ant", "Group"],
    ["Larry", "Page", "and", "Sergey", "Brin", "co-founded", "Google", "while", "at", "Stanford"],
    ["Howard", "Schultz", "expanded", "Starbucks", "into", "a", "global", "brand"],
    ["Richard", "Branson", "launched", "Virgin", "Galactic", "for", "commercial", "space", "travel"],
    ["Toyota", "introduced", "the", "Prius", "as", "the", "world's", "first", "mass-produced", "hybrid", "car"],
    ["Oprah", "Winfrey", "launched", "OWN", "and", "partnered", "with", "Weight", "Watchers"],
    ["Warren", "Buffett", "is", "the", "CEO", "of", "Berkshire", "Hathaway"],
    ["Larry", "Ellison", "co-founded", "Oracle", "Corporation"],
    ["Sony", "released", "the", "PlayStation", "5", "in", "2020"],
    ["Amazon", "introduced", "the", "Kindle", "e-reader", "in", "2007"],
    ["Elon", "Musk", "introduced", "the", "Tesla", "Model", "3", "as", "an", "affordable", "electric", "car"],
    ["IBM", "developed", "Watson", "to", "compete", "in", "AI", "and", "machine", "learning"],
    ["Apple", "introduced", "the", "MacBook", "Air", "under", "Steve", "Jobs'","leadership"],
    ["Diane", "von", "Furstenberg", "created", "the", "iconic", "wrap"]
]

# Load the Flair NER tagger
tagger = SequenceTagger.load('de-ner-large')


def extract_knowledge(sentences):
    """
    This function takes a list of tokenized sentences (each sentence is a list of tokens),
    extracts entities and relationships, and returns a list of entities and relationships.
    """

    if DEBUG:
        print("Initiating the process...")

    entities = []  # List to store all entities
    relationships = []  # List to store (Entity1, Relationship, Entity2) triples
    
    

    if DEBUG:
        print("Processing sentences...")

    for sentence in debug_sentences: # CHANGE debug_sentences to senteces
        sentence_text = " ".join(sentence)

        # Run Flair NER to extract entities
        sentence = Sentence(sentence_text)
        tagger.predict(sentence)
        local_entities = sentence.get_spans("ner")

        # Store entities uniquely in a structured way
        for entity in local_entities:
            entities.append({
                "text": entity.text,
                "label": entity.get_label("ner").value,
            })
            
    # store entities
    # Dump entities to a JSON file for evaluation
    with open('extracted_entities.json', 'w') as json_file:
        json.dump(entities, json_file, indent=4)
    

    # # Return entities and relationships for knowledge graph creation
    # return entities, relationships
    
    return




