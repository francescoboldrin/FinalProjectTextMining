"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-13 11:18:09
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-15 17:16:35
FilePath: scripts/Visualizer.py
Description: function to visualize the graph, yet to decide how
"""

import json
import pandas as pd
from TMGraph import Graph

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_to_graph(entity_file, relation_file):
    
    graph = Graph()
    
    entities = load_json(entity_file)
    for entity_data in entities:
        doc_index = entity_data["doc_index"]
        for entity_name, entity_info in entity_data["entities"].items():
            entity_type, sentence_indices = entity_info
            
            node_id = f"{doc_index}_{entity_name}"  
            graph.add_node(node_id, label=entity_type, name=entity_name, node_type=entity_type)
    
    relationships = load_json(relation_file)
    for relation_data in relationships:
        doc_index = relation_data["doc_index"]
        for relation_id, relation_info in relation_data["relationships"].items():
            entity1 = relation_info["entity1"]
            entity2 = relation_info["entity2"]
            relationship_label = relation_info["relationship"]
            sentence_index = relation_info["sentence_index"]
            
            node_id1 = f"{doc_index}_{entity1}"
            node_id2 = f"{doc_index}_{entity2}"
        
            graph.add_edge(node_id1, node_id2, label=relationship_label)
    
    return graph

entity_file = "/workspaces/FinalProjectTextMining/extracted_entities_bert_big.json"  # 替换为实际路径
relation_file = "/workspaces/FinalProjectTextMining/extracted_relationship_big.json"  # 替换为实际路径

graph = json_to_graph(entity_file, relation_file)

graph.display_graph()

df = graph.to_dataframe()
print(df)

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
for _, row in df.iterrows():
    head_node = row['head']
    tail_node = row['tail']
    relation = row['relation']
    G.add_edge(head_node, tail_node, label=relation)

pos = nx.kamada_kawai_layout(G)

plt.figure(figsize=(12, 8))  
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1500,  
    node_color='lightblue',
    font_size=8,
    font_weight='bold',
    edge_color='gray', 
    width=0.5  
)

edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6)

plt.savefig("optimized_graph.png", dpi=300, bbox_inches='tight')
plt.show()

