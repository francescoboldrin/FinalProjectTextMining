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
    
    
    
