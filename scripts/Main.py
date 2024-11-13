"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-13 11:04:33
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-13 11:22:22
FilePath: scripts/Main.py
Description: main file, loading the function that answer the assignment. when run this is the only file to be executed
"""

from DataLoader import loading_sentences
from KnowledgeExtractor import extract_knowledge
from Visualizer import visualize_graph
from Evaluator import evaluate

def main():
    sentences = loading_sentences()
    
    knownledge_graph = extract_knowledge(sentences)
    
    visualize_graph(knownledge_graph)
    
    evaluate(knownledge_graph, sentences)
    
main()