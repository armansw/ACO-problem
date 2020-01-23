import numpy as np
import time
from ACO import ACO
from MatrixGraph import MatrixGraph

mtxgraph = MatrixGraph()
distances = mtxgraph.encode_to_array('example.tsp')
aco = ACO(distances, 1, 1, 100, 0.95, alpha=1, beta=1)
mtxgraph.normalize_answer(shortest_path)
print("The shortest path in the graph is {} with length {}".format(
    shortest_path[0], shortest_path[1]))
fd.close()
