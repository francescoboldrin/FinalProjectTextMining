"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-13 11:04:33
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-15 20:53:52
FilePath: scripts/Main.py
Description: main file, loading the function that answer the assignment. when run this is the only file to be executed
"""
from flair.models import SequenceTagger
from transformers import AutoTokenizer, AutoModelForTokenClassification

from DataLoader import loading_sentences
from KnowledgeExtractor import extract_knowledge, extract_knowledge_bert
from Visualizer import visualize_graph

def main():
    documents = loading_sentences("./dataset_Linked-DocRED/train_annotated.json")

    # Load the Flair NER tagger
    # tagger = SequenceTagger.load('de-ner-large')

    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    # Load BERT-based NER pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    
    for i, document in enumerate(documents):
        print(f"elaborate doc {i}")
        knownledge_graph = extract_knowledge_bert(i, document, model, tokenizer)
        if i > 9:
            break
        
        # visualize_graph(knownledge_graph)
        # 
        # evaluate(knownledge_graph, document)
    
main()