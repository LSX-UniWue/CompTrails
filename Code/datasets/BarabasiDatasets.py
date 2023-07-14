import numpy as np
import networkx as nx

from scipy.sparse import csr_matrix
from Code.datasets.GraphWalker import PropertyWalker
from Code.datasets.AbstractDatasets import SyntheticDataset, row_wise_normalize


def create_random_graph(n: int, m: int) -> nx.Graph:
    """
    Creates a random graph with n nodes and m edges.
    """
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(n))
    edges = np.array([[(i, k) for k in np.random.choice(np.arange(n), size=m, replace=False)] for i in range(n)]).reshape(-1, 2)
    graph.add_edges_from(edges)
    return graph

class ReducedBarabasiAlbertDataset(SyntheticDataset):
    def __init__(self, args: dict):
        super().__init__(args=args)
        self.amount_walks = args['amount_walks']
        self.max_walk_len = args['max_walk_len']
        self.max_importance = args['max_importance'] if 'max_importance' in args else None
        self.use_importance = args['use_importance']
        self.transition_probas = args['transition_probas']
        for i, transition_proba in enumerate(self.transition_probas):
            if sum(transition_proba.values()) != 1.0:
                print("ERROR: Transition probabilities of graph ", i, " do not sum up to 1. ")
                exit(1)
        self.graphs = [nx.barabasi_albert_graph(size, transition_counts) for size, transition_counts in zip(self.sizes, self.max_transition_count)]
        if self.max_importance:
            # Adding Importance scores to each node in both graphs (between 0 and 10)
            labels = [{node: np.random.randint(self.max_importance) for node in graph.nodes} for graph in self.graphs]
            [nx.set_node_attributes(self.graphs[i], labels[i], "Importance") for i in range(len(self.graphs))]
            print_importance_scores = [(i, len([x for x in labels[0].values() if x == i])) for i in range(self.max_importance)]
            print(f"Having the following importance scores for graph 0: {print_importance_scores}")
        self.walkers = [PropertyWalker(graph, transition_probs=self.transition_probas[i],
                                       max_walk_len=self.max_walk_len[i], amount_walks=self.amount_walks[i])
                        for i, graph in enumerate(self.graphs)]
        self.choices = [{"average": walker.get_average_choice(), "uniform": walker.get_uniform_choice()}
                        for walker in self.walkers]
        if self.use_importance:
            print("Reducing based on importance. ")
            self.transitions = [self.reduce_transitions_by_importance(self.walkers[i].get_transitions(), self.graphs[i])
                                for i in range(len(self.transition_probas))]
            hyps = {f"importance-{i}": csr_matrix(np.array([np.eye(self.max_importance)[i] for _ in np.arange(self.max_importance)])) for i in range(self.max_importance)}
            hyps["uni"] = csr_matrix(np.ones((self.max_importance, self.max_importance)) / np.ones((self.max_importance, self.max_importance)).sum(axis=1))
            self.hypotheses = [hyps for _ in range(len(self.sizes))]
        else:
            print("Reducing based on even/odd. ")
            self.transitions = [self.reduce_transitions(csr_matrix(self.walkers[i].get_transitions())) for i in range(len(self.transition_probas))]
            self.hypotheses = [{
                "even": csr_matrix(np.array([np.eye(2)[0] for _ in np.arange(2)])),
                "odd": csr_matrix(np.array([np.eye(2)[1] for _ in np.arange(2)[::-1]])),
                "uni": csr_matrix(np.ones((2, 2)) / np.ones((2, 2)).sum(axis=1))
            } for _ in range(len(self.sizes))]

    def reduce_transitions_by_importance(self, transitions: csr_matrix, graph: nx.Graph) -> csr_matrix:
        reduced_adjacency_matrix = csr_matrix((self.max_importance, self.max_importance))
        for idy, row in enumerate(transitions):
            curr_importance_score = graph.nodes[idy]['Importance']
            importance_scores = [graph.nodes[x]['Importance'] for x in row.indices]
            for importance, value in zip(importance_scores, row.data):
                reduced_adjacency_matrix[curr_importance_score, importance] += value
        return reduced_adjacency_matrix

    @staticmethod
    def reduce_transitions(transitions: csr_matrix) -> csr_matrix:
        reduced_adjacency_matrix = np.zeros(4).reshape((2, 2))
        for idy, row in enumerate(transitions):
            to_even = sum([data for data, idy in zip(row.data, row.indices) if idy % 2 == 0])
            to_odd = sum([data for data, idy in zip(row.data, row.indices) if idy % 2 == 1])
            if idy % 2 == 0:
                reduced_adjacency_matrix[0, 0] += to_even
                reduced_adjacency_matrix[0, 1] += to_odd
            else:  # node % 2 == 1
                reduced_adjacency_matrix[1, 0] += to_even
                reduced_adjacency_matrix[1, 1] += to_odd
        return csr_matrix(reduced_adjacency_matrix)

    def get_transitions(self) -> list:
        return self.transitions

    def get_hypotheses(self) -> list:
        return self.hypotheses

    def get_choices(self):
        return self.choices


class BarabasiAlbertDataset(SyntheticDataset):
    def __init__(self, args):
        super().__init__(args)
        self.amount_walks = args['amount_walks']
        self.max_walk_len = args['max_walk_len']
        self.max_importance = args['max_importance']
        self.transition_probas = args['transition_probas']
        for i, transition_proba in enumerate(self.transition_probas):
            if sum(transition_proba.values()) != 1.0:
                print("ERROR: Transition probabilities of graph ", i, " do not sum up to 1. ")
                exit(1)
        print("Transition probabilities are: ", self.transition_probas)
        self.actual_transition_counts = []
        for counts, sizes in zip(self.max_transition_count, self.sizes):
            if isinstance(counts, int):
                self.actual_transition_counts.append(counts)
            else:
                self.actual_transition_counts.append(int(counts * sizes))
        if args['graph_type'] == "random":
            self.graphs = [create_random_graph(size, transition_counts) for size, transition_counts in zip(self.sizes, self.actual_transition_counts)]
        else:  # args['graph_type'] == "barabasi"
            self.graphs = [nx.barabasi_albert_graph(size, transition_counts) for size, transition_counts in zip(self.sizes, self.actual_transition_counts)]
        # Adding Importance scores to each node in both graphs (between 0 and 10)
        labels = [{node: np.random.randint(self.max_importance) for node in graph.nodes} for graph in self.graphs]
        [nx.set_node_attributes(self.graphs[i], labels[i], "Importance") for i in range(len(self.graphs))]
        print(f"\tGraph props: \n"
              f"\t Nodes: {[len(x.nodes) for x in self.graphs]} \n"
              f"\t Edges: {[len(x.edges) for x in self.graphs]} \n "
              f"\t Degree: Min - {[np.array(x.degree)[:, 1].min() for x in self.graphs]} / "
              f"Max - {[np.array(x.degree)[:, 1].max() for x in self.graphs]} / "
              f"Mean - {[np.array(x.degree)[:, 1].mean() for x in self.graphs]} \n "
              f"\t Density: {[nx.density(x) for x in self.graphs]}")
        self.walkers = [PropertyWalker(graph, transition_probs=self.transition_probas[i],
                                       max_walk_len=self.max_walk_len[i], amount_walks=self.amount_walks[i],
                                       max_importance=self.max_importance)
                        for i, graph in enumerate(self.graphs)]
        self.choices = [{"average": walker.get_average_choice(), "uniform": walker.get_uniform_choice()} for walker in self.walkers]
        self.transitions = [csr_matrix(self.walkers[i].get_transitions()) for i in range(len(self.transition_probas))]
        self.hypotheses = []
        for idx in range(len(self.graphs)):
            self.hypotheses.append(self.create_hypothesis(graph=self.graphs[idx], shape=self.transitions[idx].shape))

    def create_hypothesis(self, graph, shape) -> dict:
        even = np.zeros(shape)
        even[:, [x for x in range(len(even)) if x % 2 == 0]] = 1
        odd = np.zeros(shape)
        odd[:, [x for x in range(len(odd)) if x % 2 == 1]] = 1
        hub = csr_matrix(shape)
        descend = csr_matrix(shape)
        most_important = csr_matrix(shape)
        least_important = csr_matrix(shape)
        prop_imp = csr_matrix(shape)
        anti_prop_imp = csr_matrix(shape)
        high_five_flat = csr_matrix(shape)
        high_five_dist = csr_matrix(shape)
        for node in graph:
            degree_order = sorted([(x, graph.degree[x]) for x in graph.neighbors(node)], key=lambda x: x[1])
            hub[node, degree_order[-1][0]] = 1
            descend[node, degree_order[0][0]] = 1
            neighbors = list(graph.neighbors(node))
            importance_scores = np.array([graph.nodes[x]["Importance"] for x in neighbors])
            # do most important
            temp = np.argwhere(importance_scores == np.max(importance_scores)).reshape(-1)
            for tmp in temp:
                most_important[node, neighbors[tmp]] = 1 / len(temp)
            # do least important
            temp = np.argwhere(importance_scores == np.min(importance_scores)).reshape(-1)
            for tmp in temp:
                least_important[node, neighbors[tmp]] = 1 / len(temp)
            # do high 5 flat
            temp = np.argwhere(importance_scores > 4).reshape(-1)
            for tmp in temp:
                high_five_flat[node, neighbors[tmp]] = 1 / len(temp)
            high5_imp_scores = importance_scores * np.where(importance_scores > 4, 1, 0)
            for target, curr_array in [(prop_imp, importance_scores),
                                       (anti_prop_imp, self.max_importance - np.array(importance_scores)),
                                       (high_five_dist, high5_imp_scores)]:
                tmp_sum = sum(curr_array)
                for neighbor_id, importance_prob in zip(neighbors, curr_array):
                    target[node, neighbor_id] = importance_prob / tmp_sum
        adjacency_matrix = nx.adjacency_matrix(graph).todense()
        return {"teleport": row_wise_normalize(np.ones(shape)),
                "even_graph": row_wise_normalize(np.multiply(even, adjacency_matrix)),
                "odd_graph": row_wise_normalize(np.multiply(odd, adjacency_matrix)),
                "uni": row_wise_normalize(np.multiply(np.ones(shape), adjacency_matrix)),
                "even": row_wise_normalize(even), "odd": row_wise_normalize(odd),
                "hub": hub, "desc": descend,
                "high_5_flat": high_five_flat, "high_5_dist": high_five_dist,
                "only_most_important": most_important, "only_least_important": least_important,
                "prop_imp": prop_imp, "anti_prop_imp": anti_prop_imp, 
                "noise-25": self.add_systematic_noise(adjacency_matrix, 0.25)
                }

    @staticmethod
    def add_systematic_noise(adjacency_matrix: np.array, noise: float) -> csr_matrix:
        ret_val = np.zeros(adjacency_matrix.shape)
        for i in range(len(adjacency_matrix)):
            if adjacency_matrix[i].sum() != 0:
                scaling = adjacency_matrix[i].max() / 2
                for j in range(len(adjacency_matrix[i])):
                    diff = 2 * noise * (scaling - adjacency_matrix[i, j])
                    ret_val[i, j] = adjacency_matrix[i, j] + diff
            ret_val[i] /= ret_val[i].sum()
        return csr_matrix(ret_val)

    def get_transitions(self) -> list:
        return self.transitions

    def get_hypotheses(self) -> list:
        return self.hypotheses

    def get_choices(self):
        return self.choices