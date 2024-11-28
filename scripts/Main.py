"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-13 11:04:33
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-25 11:41:46
FilePath: scripts/Main.py
Description: main file, loading the function that answer the assignment. when run this is the only file to be executed
"""
from flair.models import SequenceTagger
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from DataLoader import loading_sentences
from KnowledgeExtractor import extract_knowledge, extract_knowledge_bert, entity_extraction_llm, extract_relationship_llm
from relation_extraction import rebel_large_relation_extraction
# from Visualizer import visualize_graph
import os
import google.generativeai as genai


def main():
    documents = loading_sentences("./dataset_Linked-DocRED/train_annotated.json")

    # Load the Flair NER tagger
    # tagger = SequenceTagger.load('de-ner-large')

    # model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    # # Load BERT-based NER pipeline
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForTokenClassification.from_pretrained(model_name)

    # triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large',tokenizer='Babelscape/rebel-large')

    genai.configure(api_key="AIzaSyB7aL_szQfU-PX-JBC0MyJWxA0Y5_44Hpk")
    
    
    for i, document in enumerate(documents):
        # print(f"elaborate doc {i}")
        # knownledge_graph = extract_knowledge_bert(i, document, model, tokenizer)
        
        entity_extraction_llm(i, document, genai)
        
        
        
        if i > 9:
            break
        
        # visualize_graph(knownledge_graph)
        # 
        # evaluate(knownledge_graph, document)
    
main()