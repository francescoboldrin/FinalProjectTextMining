"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-13 11:04:33
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-28 19:18:52
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
from Evaluator import evaluate_ner, evaluate_relationship_extraction


def main():
    """
    Main function that runs the pipeline
    
    :return: None
    
    it loads the documents, extract the knowledge and the relationships 
    """
    
    documents = loading_sentences("./dataset_Linked-DocRED/train_annotated.json")

    # POSSIBLE MODELS:
    # Load the Flair NER tagger
    # tagger = SequenceTagger.load('de-ner-large')

    # model_name = "dbmdz/bert-large-cased-finetuned-conll03-english" or "dslim/bert-base-NER"
    # # Load BERT-based NER pipeline
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForTokenClassification.from_pretrained(model_name)

    # triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large',tokenizer='Babelscape/rebel-large')

    # genai.configure(api_key="GEMINI_API_KEY")
    
    
    for i, document in enumerate(documents):
        print("Processing document", i)
        
        # POSSIBLE APPROACHES:
        # Extract knowledge using Flair NER tagger not pre-trained
        # extract_knowledge(document, tagger)
        
        # Extract knowledge using BERT-based NER
        # extract_knowledge_bert(document, tokenizer, model)
        
        # Extract entities and relationships using LLM
        # entity_extraction_llm(document, genai)
        # extract_relationship_llm(document, genai)
        
        # Extract relationships using REBEL
        # rebel_large_relation_extraction(i, document, triplet_extractor)
    
        # POSSIBLE EVALUATIONS:
        # Evaluate NER
        # evaluate_ner(path_to_gt, path_to_extracted)
        
        # Evaluate relationship extraction
        # evaluate_relationship_extraction(path_to_gt, path_to_extracted)
        
    
main()