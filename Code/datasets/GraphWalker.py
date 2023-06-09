import networkx as nx

import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

from abc import ABC, abstractmethod


class GraphWalker(ABC):
    def __init__(self, graph: nx.MultiGraph, max_walk_len: int, amount_walks: int):
        self.graph = graph
        self.max_walk_len = max_walk_len
        self.amount_walks = amount_walks
        self.walks = []

    @abstractmethod
    def walk(self, start_node: int):
        pass

    @abstractmethod
    def get_transitions(self):
        pass

    @abstractmethod
    def create_matrix(self):
        pass


class PropertyWalker(GraphWalker):
    def __init__(self, graph: nx.MultiGraph, transition_probs: dict, max_walk_len: int, amount_walks: int = 1, max_importance: int = 10, verbose: int = 0):
        super().__init__(graph, max_walk_len, amount_walks)
        if verbose:
            print(f"Running Property Walker with graph size {len(graph)}. ")
        self.transition_probs = transition_probs
        self.max_importance = max_importance
        self.choices = []
        self.amount_uniform = 0
        self.overall_current_importance_score = 0
        for node in tqdm(graph.nodes, desc="Running nodes on Property walker ..."):
            for _ in range(self.amount_walks):
                self.walks.append(self.walk(node))
        self.transitions = self.create_matrix()
        self.average_choice = list(self.transition_probs.keys())[round(sum(self.choices) / len(self.choices))]
        print(f"\tBiased random walker: Choices - {len(self.choices)} / "
              f"Average - {sum(self.choices) / len(self.choices)} ({self.average_choice}) / "
              f"Uniform fallback: {self.amount_uniform / len(self.choices)}")

    def walk(self, start_node: int):
        walk = [start_node]
        while len(walk) < self.max_walk_len:
            adjacent_nodes = list(self.graph.neighbors(walk[-1]))
            behaviour_probas = np.fromiter(self.transition_probs.values(), dtype=float)
            choice = np.random.choice(behaviour_probas.shape[0], 1, p=behaviour_probas)[0]
            self.choices.append(choice)
            transition_behaviour = list(self.transition_probs.keys())[choice]
            if transition_behaviour == "even":
                curr_prob = np.array([1 if x % 2 == 0 else 0 for x in adjacent_nodes])
                if sum(curr_prob) == 0:
                    curr_prob = np.ones(len(curr_prob)) / len(curr_prob)
                    self.amount_uniform += 1
                probas = curr_prob / curr_prob.sum()
            elif transition_behaviour == "odd":
                curr_prob = np.array([1 if x % 2 == 1 else 0 for x in adjacent_nodes])
                if sum(curr_prob) == 0:
                    curr_prob = np.ones(len(curr_prob)) / len(curr_prob)
                    self.amount_uniform += 1
                probas = curr_prob / curr_prob.sum()
            elif transition_behaviour == "only_most_important":
                importance_scores = [self.graph.nodes[x]["Importance"] for x in adjacent_nodes]
                curr_prob = np.where(importance_scores == np.max(importance_scores), 1.0, 0.0)
                probas = curr_prob / curr_prob.sum()
            elif transition_behaviour == "only_least_important":
                importance_scores = [self.graph.nodes[x]["Importance"] for x in adjacent_nodes]
                curr_prob = np.where(importance_scores == np.min(importance_scores), 1.0, 0.0)
                probas = curr_prob / curr_prob.sum()
            elif "prop_imp" in transition_behaviour:
                if transition_behaviour == "prop_imp":
                    curr_scores = np.array([self.graph.nodes[x]["Importance"] for x in adjacent_nodes])
                elif transition_behaviour == "anti_prop_imp":
                    curr_scores = [self.graph.nodes[x]["Importance"] for x in adjacent_nodes]
                    curr_scores = np.array([x + 2 * ((self.max_importance / 2) - x) for x in curr_scores])
                elif transition_behaviour == "prop_imp_5":
                    curr_scores = np.array([self.graph.nodes[x]["Importance"] for x in adjacent_nodes])
                    curr_scores = curr_scores * np.where(curr_scores > 4, 1.0, 0.0)
                elif transition_behaviour == "anti_prop_imp_5":
                    curr_scores = np.array([self.graph.nodes[x]["Importance"] for x in adjacent_nodes])
                    curr_scores = curr_scores * np.where(curr_scores > 4, 1.0, 0.0)
                    curr_scores = np.array([x + 2 * ((self.max_importance / 2) - x) for x in curr_scores])
                else:
                    print(f"ERROR: Cannot find {transition_behaviour}. ")
                    exit(1)
                if sum(curr_scores) == 0:
                    probas = np.ones(len(curr_scores)) / len(curr_scores)
                    self.amount_uniform += 1
                else:
                    probas = curr_scores / sum(curr_scores)
            elif transition_behaviour == "anti_prop_imp":
                rev_importance_scores = [self.graph.nodes[x]["Importance"] for x in adjacent_nodes]
                rev_importance_scores = np.array([x + 2 * ((self.max_importance / 2) - x) for x in rev_importance_scores])
                if sum(rev_importance_scores) == 0:
                    probas = np.ones(len(rev_importance_scores)) / len(rev_importance_scores)
                    self.amount_uniform += 1
                else:
                    probas = rev_importance_scores / sum(rev_importance_scores)
            elif transition_behaviour == "hub":
                curr_prob = np.array([self.graph.degree[x] for x in adjacent_nodes])
                if sum(curr_prob) == 0:
                    curr_prob = np.ones(len(curr_prob)) / len(curr_prob)
                    self.amount_uniform += 1
                probas = curr_prob / curr_prob.sum()
            elif transition_behaviour == "descend":
                curr_prob = np.array([self.graph.degree[x] for x in adjacent_nodes])
                if sum(curr_prob) == 0:
                    curr_prob = np.ones(len(curr_prob)) / len(curr_prob)
                    self.amount_uniform += 1
                curr_prob = max(curr_prob) - curr_prob
                probas = curr_prob / curr_prob.sum()
            else:  # will be uniform
                curr_prob = np.array([1 for _ in adjacent_nodes])
                probas = curr_prob / curr_prob.sum()
                self.amount_uniform += 1
            walk.append(np.random.choice(adjacent_nodes, 1, p=probas)[0])
            if "imp" in transition_behaviour:
                self.overall_current_importance_score += self.graph.nodes[walk[-1]]['Importance']
        return walk

    def get_transitions(self) -> csr_matrix:
        return self.transitions

    def create_matrix(self) -> csr_matrix:
        matrix = csr_matrix((len(self.graph.nodes), len(self.graph.nodes)))
        for walk in self.walks:
            for idx in range(len(walk) - 1):
                matrix[walk[idx], walk[idx + 1]] += 1
        return matrix

    def get_average_choice(self):
        return self.average_choice

    def get_uniform_choice(self):
        return self.amount_uniform / len(self.choices)


class UniformWalker(GraphWalker):
    def __init__(self, graph: nx.MultiGraph, max_walk_len: int, amount_walks: int = 1, verbose: int = 1):
        super().__init__(graph, max_walk_len, amount_walks)
        if verbose:
            print(f"Running Uniform Walke for graph size {len(graph)}.")
        for node in tqdm(graph.nodes, desc="Running nodes on UniformWalker ..."):
            for _ in range(self.amount_walks):
                self.walks.append(self.walk(node))
        self.transitions = self.create_matrix()

    def walk(self, start_node: int) -> list:
        walk = [start_node]
        while len(walk) < self.max_walk_len:
            adjacent_nodes = list(self.graph.neighbors(walk[-1]))
            walk.append(np.random.choice(adjacent_nodes, 1)[0])
        return walk

    def get_transitions(self):
        return self.transitions

    def create_matrix(self) -> np.array:
        matrix = np.zeros((len(self.graph.nodes), len(self.graph.nodes)))
        for walk in self.walks:
            for idx in range(len(walk) - 1):
                matrix[walk[idx], walk[idx + 1]] += 1
        return csr_matrix(matrix)


class HubWalker(GraphWalker):
    def __init__(self, graph: nx.MultiGraph,
                 max_walk_len: int,
                 amount_walks: int = 1,
                 factor: float = 1.0,
                 reverse: bool = False):
        super().__init__(graph, max_walk_len, amount_walks)
        self.factor = factor
        self.reverse = reverse
        for node in graph.nodes:
            for _ in range(self.amount_walks):
                self.walks.append(self.walk(node))
        self.transitions = self.create_matrix()

    def walk(self, start_node: int) -> list:
        walk = [start_node]
        while len(walk) < self.max_walk_len:
            adjacent_nodes = list(self.graph.neighbors(walk[-1]))
            adjacent_degrees = np.array([self.graph.degree[x] for x in adjacent_nodes])
            if self.reverse:
                adjacent_degrees = max(adjacent_degrees) - np.array(adjacent_degrees)
            prob = adjacent_degrees / adjacent_degrees.sum()
            if self.factor != 1:
                uniform = np.ones(len(prob)) / len(prob)
                prob = np.mean([prob, uniform], axis=0)
            if prob.sum() == 0:
                prob = np.ones(len(prob)) / prob.sum()
            walk.append(np.random.choice(adjacent_nodes, 1, p=prob)[0])
        return walk

    def get_transitions(self):
        return self.transitions

    def create_matrix(self) -> np.array:
        matrix = np.zeros((len(self.graph.nodes), len(self.graph.nodes)))
        for walk in self.walks:
            for idx in range(len(walk) - 1):
                matrix[walk[idx], walk[idx + 1]] += 1
        return matrix


class EvenNodeWalker(GraphWalker):
    def __init__(self, graph: nx.MultiGraph, max_walk_len: int, amount_walks: int = 1, even: bool = True, verbose: int = 0):
        super().__init__(graph, max_walk_len, amount_walks)
        if verbose:
            print(f"Running Even Node Walker for graph size {len(graph)}")
        self.even = 0 if even else 1
        for node in tqdm(graph.nodes, desc="Running nodes on EvenNodeWalker ..."):
            for _ in range(self.amount_walks):
                self.walks.append(self.walk(node))
        self.transitions = self.create_matrix()

    def walk(self, start_node: int) -> list:
        walk = [start_node]
        while len(walk) < self.max_walk_len:
            adjacent_nodes = list(self.graph.neighbors(walk[-1]))
            prob = [1 if x % 2 == self.even else 0 for x in adjacent_nodes]
            if sum(prob) == 0:
                prob = np.ones(len(prob)) / len(prob)
            else:
                prob = np.array(prob) / sum(prob)
            walk.append(np.random.choice(adjacent_nodes, 1, p=prob)[0])
        return walk

    def get_transitions(self):
        return self.transitions

    def create_matrix(self) -> np.array:
        matrix = np.zeros((len(self.graph.nodes), len(self.graph.nodes)))
        for walk in self.walks:
            for idx in range(len(walk) - 1):
                matrix[walk[idx], walk[idx + 1]] += 1
        return matrix


class SameNodeWalker(GraphWalker):
    def __init__(self, graph: nx.MultiGraph, max_walk_len: int, amount_walks: int = 1, verbose: int = 0):
        super().__init__(graph, max_walk_len, amount_walks)
        self.uniform = 0
        if verbose:
            print(f"Running Same Node Walker for graph size {len(graph)}")
        for node in tqdm(graph.nodes, desc="Running nodes on SameNodeWalker ..."):
            for _ in range(self.amount_walks):
                self.walks.append(self.walk(node))
        self.transitions = self.create_matrix()
        print(f"Having {self.uniform} uniform decisions. ")

    def walk(self, start_node: int) -> list:
        walk = [start_node]
        while len(walk) < self.max_walk_len:
            adjacent_nodes = list(self.graph.neighbors(walk[-1]))
            even = walk[-1] % 2 == 0 
            prob = [1 if x % 2 != even else 0 for x in adjacent_nodes]
            if sum(prob) == 0:
                prob = np.ones(len(prob)) / len(prob)
                self.uniform += 1
            else:
                prob = np.array(prob) / sum(prob)
            walk.append(np.random.choice(adjacent_nodes, 1, p=prob)[0])
        return walk

    def get_transitions(self):
        return self.transitions

    def create_matrix(self) -> np.array:
        matrix = np.zeros((len(self.graph.nodes), len(self.graph.nodes)))
        for walk in self.walks:
            for idx in range(len(walk) - 1):
                matrix[walk[idx], walk[idx + 1]] += 1
        return csr_matrix(matrix)
