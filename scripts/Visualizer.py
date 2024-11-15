"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-13 11:18:09
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-13 11:22:22
FilePath: scripts/Visualizer.py
Description: function to visualize the graph, yet to decide how
"""

def visualize_graph(graph):
    """
    Description:
    
    """
    
    return


graph = Graph()
graph.add_node(1, 'Person', 'Alice', 'Person')
graph.add_node(2, 'Person', 'Bob', 'Person')
graph.add_node(3, 'Company', 'OpenAI', 'Company')
graph.add_node(4, 'Product', 'GPT-3', 'Product')
graph.add_node(5, 'Product', 'ChatGPT', 'Product')
graph.add_node(6, 'Company', 'Tesla', 'Company')
graph.add_node(7, 'Person', 'Elon Musk', 'Person')

graph.add_edge(1, 2, 'knows')
graph.add_edge(1, 3, 'works_at')
graph.add_edge(1, 4, 'uses')
graph.add_edge(2, 5, 'uses')
graph.add_edge(3, 4, 'develops')
graph.add_edge(3, 5, 'develops')
graph.add_edge(7, 6, 'founded')
graph.add_edge(2, 6, 'interested_in')

graph.display_graph()

df = graph.to_dataframe()
print("\nDataFrame representation of the Graph:")
print(df)

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

for _, row in df.iterrows():
    head_node = row['head']  
    tail_node = row['tail']  
    relation = row['relation']
 
    G.add_edge(head_node, tail_node, label=relation)

pos = nx.spring_layout(G)
labels = nx.get_edge_attributes(G, 'label')

nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=5, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')

plt.show()
