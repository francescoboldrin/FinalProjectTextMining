"""
Author: francesco boldrin francesco.boldrin@studenti.unitn.it
Date: 2024-11-13 14:27:30
LastEditors: francesco boldrin francesco.boldrin@studenti.unitn.it
LastEditTime: 2024-11-13 14:32:10
FilePath: utils/TMGraph.py
Description: Contains the definition of the class Graph
"""

"""
nodes = {
            id: label (name, type)
            ...
        }
edges = [[id1,id2, label],...]
"""

import pandas as pd

class Graph:
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.num_of_nodes = 0
        self.num_of_edges = 0
    
    def add_node(self, node_id, label, name, node_type):
    
        self.nodes[node_id] = {'label': label, 'name': name, 'type': node_type}
        self.num_of_nodes += 1
    
    def add_edge(self, id1, id2, label):
        self.edges.append([id1, id2, label])
        self.num_of_edges += 1
    
    def display_graph(self):
       
        print("Nodes:")
        for node_id, properties in self.nodes.items():
            print(f"  {node_id}: {{'name': '{properties['name']}', 'type': '{properties['type']}'}}")
        
        print("\nEdges:")
        for edge in self.edges:
            print(f"  {edge[0]} -- {edge[2]} --> {edge[1]}")
    
    def to_dataframe(self):
        head = []
        relation = []
        tail = []
        
        for edge in self.edges:
            
            head.append((self.nodes[edge[0]]['name'], self.nodes[edge[0]]['type']))  
            relation.append(edge[2])  
            tail.append((self.nodes[edge[1]]['name'], self.nodes[edge[1]]['type']))  
    
        df = pd.DataFrame({'head': head, 'relation': relation, 'tail': tail})
        return df

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


    
    
    
