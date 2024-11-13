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

class Graph:
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.num_of_nodes = 0
        self.num_of_edges = 0
    
    def add_node(self, node_id, label, name, node_type):
        #
        self.nodes[node_id] = {'label': label, 'name': name, 'type': node_type}
        self.num_of_nodes += 1
    
    def add_edge(self, id1, id2, label):
        
        self.edges.append([id1, id2, label])
        self.num_of_edges += 1
    
    def display_graph(self):
       
        print("Nodes:")
        for node_id, properties in self.nodes.items():
            print(f"  {node_id}: {properties}")
        
        print("\nEdges:")
        for edge in self.edges:
            print(f"  {edge[0]} -- {edge[2]} --> {edge[1]}")
    
    
    
