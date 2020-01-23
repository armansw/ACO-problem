import random as rn
import numpy as np
from numpy.random import choice as np_choice
from mpi4py import MPI
import math
import time

comm = MPI.COMM_WORLD



class ACO(object):
    def __init__(self, distances, ants_num, best_ant, iterations_num, pheromone_rate, alpha=1, beta=1):
        """
        Args:
            distance (2d np array) - square matrix of distances. Diagonal elements assumed to be np.inf
            ants_num (natural) - number of ants running each iteration
            best_ant (natural) - number of best ants who deposit pheromone
            iterations_num (natural) - number of iterations
            decay (float) - Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
        """

        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.ants_num = ants_num
        self.best_ant = best_ant
        self.iterations_num = iterations_num
        self.pheromone_rate = pheromone_rate
        self.alpha = alpha
        self.beta = beta

    def spread_pheromone(self, all_paths, best_ant, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:best_ant]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for node in path:
            total_dist += self.distances[node]
        return total_dist

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * ((1.0 / dist)**self.beta)
        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(
                self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))
        return path

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.ants_num):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def run(self):
        processes_num = comm.Get_size()
        rank = comm.Get_rank()
        shortest_path = None
        if(rank == 0):
            all_time_shortest_path = ("placeholder", np.inf)
            for i in range(self.iterations_num):
                for j in range(1, processes_num):
                    shortest_path = comm.recv(source=j, tag=j)
                    if(shortest_path[1] < all_time_shortest_path[1]):
                        all_time_shortest_path = shortest_path
                    self.pheromone *= self.pheromone_rate
            return all_time_shortest_path
        else:
            for i in range(self.iterations_num):
                all_paths = self.gen_all_paths()
                self.spread_pheromone(all_paths, self.best_ant,
                                      shortest_path=shortest_path)
                shortest_path = min(all_paths, key=lambda x: x[1])
                comm.send(shortest_path, dest=0, tag=rank)

