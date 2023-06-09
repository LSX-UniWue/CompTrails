import sys
import numpy as np
import math
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from scipy.sparse import csr_matrix
from pytrails.hyptrails import MarkovChain
from scipy.stats import sem
from scipy.spatial.distance import jensenshannon
import networkx as nx
from scipy.stats import mannwhitneyu
from Code.datasets.GraphWalker import SameNodeWalker, UniformWalker


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 1.6

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def cohend(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / s


class SyntheticDataset:
    def __init__(self, args: dict) -> None:
        self.is_sampling = args['is_sampling']
        self.sample_size = args['sample_size']
        self.sizes = args['sizes'] 
        self.transition_counts = args['transition_counts']
        self.walk_lengths = args['walk_lengths']
        self.walks_per_node = args['walks_per_node']
        self.state_sample_method = args['state_sample_method']
        self.mixtures = args['mixtures']
        self.hyptrails_ks = args['hyptrails_ks']
        self.runs = args['runs']
        if args['data_generation'] == 'random':
            self.hypotheses = [self.get_normalized_random_matrix(size) for size in self.sizes]
            self.transitions = [self.create_transitions(i) for i in range(len(self.sizes))]
            for i, (transition, hypothesis) in enumerate(zip(self.transitions, self.hypotheses)):
                print(f"Network {i}: Transitions-{transition.sum()} Hypothesis-{hypothesis.sum()}")
        elif args['data_generation'] == 'barabasi':
            self.graphs = [nx.barabasi_albert_graph(size, counts) for size, counts in zip(self.sizes, self.transition_counts)]
            # transition_probs = [{'even': 1.0-i, 'odd': i} for i in self.mixtures]
            walker = [UniformWalker if mix == 0.5 else SameNodeWalker for mix in self.mixtures]
            # todo instead of property walker, use same kinded walker and uniform walker 
            self.transition_walkers = [walker[i](graph=graph, max_walk_len=self.walk_lengths[i], amount_walks=self.walks_per_node, verbose=1) for i, graph in enumerate(self.graphs)]
            self.transitions = [transition_walker.get_transitions().A for transition_walker in self.transition_walkers]
            self.hypotheses = [self.create_same_kinded_hypothesis(size=size, transition=transition) for size, transition in zip(self.sizes, self.transitions)]
            # self.hypotheses_walkers = [SameNodeWalker(graph=graph, max_walk_len=self.walk_lengths[i], amount_walks=self.walks_per_node, verbose=1) for i, graph in enumerate(self.graphs)]
            # self.hypotheses = [hypotheses_walker.get_transitions() for hypotheses_walker in self.hypotheses_walkers]
            for i in range(len(self.graphs)):
                print(f"Graph {i}: Nodes-{len(self.graphs[i])}, Edges-{self.graphs[i].size()} Transitions-{self.transitions[i].sum()} Hypothesis-{self.hypotheses[i].sum()}")
        else:
            print(f"ERROR: Data generation method {args['data_generation']} not found. ")
            sys.exit(1)
        self.hyptrails_evidence = self.calculate_hyptrails()
        self.js_divergence = self.calculate_js_divergence()
            
    @staticmethod
    def get_normalized_random_matrix(size: int) -> np.ndarray:
        matrix = np.random.rand(size, size)
        for row in matrix:
            row /= row.sum()
        return matrix

    def create_transitions(self, i: int) -> np.ndarray:
        new_random = self.get_normalized_random_matrix(self.sizes[i])
        transitions = ((1 - self.mixtures[i]) * new_random + self.mixtures[i] * self.hypotheses[i]) * max(self.sizes)
        return transitions.astype(int)

    @staticmethod
    def create_same_kinded_hypothesis(size: int, transition: np.ndarray) -> np.ndarray:
        ret_matrix = np.zeros((size, size))
        for row in range(size):
            for col in range(size):
                if row != col:
                    if (row % 2 == 0 and col % 2 == 0) or (row % 2 == 1 and col % 2 == 1):
                        ret_matrix[row, col] = 1
        # mask = transition.copy()
        # mask[mask != 0] = 1
        # ret_matrix = ret_matrix * mask
        return ret_matrix


    def sample_states(self, transitions: np.ndarray, hypothesis: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.state_sample_method == "random":
            sampled_states = np.random.randint(transitions.shape[0], size=size)
        elif self.state_sample_method == "snowball":
            sampled_states = [np.random.randint(transitions.shape[0])]
            new_seeds = set(sampled_states)
            while len(sampled_states) < size:
                neighbors = set()
                for seed in new_seeds:
                    neighbors.update(np.nonzero(transitions[seed])[-1])
                new_seeds = neighbors - new_seeds
                nodes_left = size - len(sampled_states)
                if len(new_seeds) >= nodes_left:
                    sampled_states.extend(np.random.choice(list(new_seeds), nodes_left))
                elif len(new_seeds) == 0:  # restart
                    new_seeds = {np.random.randint(transitions.shape[0])}
                    sampled_states.extend(new_seeds)
                else:
                    sampled_states.extend(neighbors)
                    new_seeds = neighbors
        else:
            print(f"ERROR: Cannot find state sample method {self.state_sample_method}.")
            sys.exit(1)
        if len(sampled_states) != size:
            print(f"ERROR: Sampled {len(sampled_states)} instead of {size}. ")
            sampled_states = sampled_states[:size]
        sample_transitions = transitions[:, sampled_states][sampled_states, :]
        sample_hyp = hypothesis[:, sampled_states][sampled_states, :]
        for row in sample_hyp:
            if row.sum() != 0:
                row /= row.sum()
        return sample_transitions, sample_hyp

    @staticmethod
    def sample_transitions(transitions: list, size: int, normalise: bool = False) -> list:
        sampled_transitions = [[] for _ in range(len(transitions))]
        for row in range(transitions[0].shape[0]):
            sample_size = int(max([x[row].sum() for x in transitions]))
            for transition_idx in range(len(transitions)):
                if transitions[transition_idx][row].sum() == 0:
                    prob_distr = np.ones(transitions[transition_idx][row].shape[0]) / transitions[transition_idx][row].shape[0]
                else:
                    prob_distr = transitions[transition_idx][row] / transitions[transition_idx][row].sum()
                samples = np.random.choice(size, size=sample_size, p=prob_distr)
                reduce_by_key = np.array([len(np.where(samples == x)[0]) for x in range(size)])
                if normalise:
                    reduce_by_key = reduce_by_key / reduce_by_key.sum()
                sampled_transitions[transition_idx].append(reduce_by_key)
        sampled_transitions = [np.array(x) for x in sampled_transitions]
        return sampled_transitions

    def calculate_hyptrails(self) -> Tuple[np.ndarray, np.ndarray]:
        evidence = []
        if self.is_sampling:
            for _ in tqdm(range(self.runs), desc="Collecting evidence. "):
                curr_evidence = []
                iter_trans, iter_hyp = [], []
                for transitions, hypothesis in zip(self.transitions, self.hypotheses):
                    curr_transitions, curr_hypothesis = self.sample_states(transitions=transitions, hypothesis=hypothesis, size=self.sample_size)
                    iter_trans.append(curr_transitions)
                    iter_hyp.append(curr_hypothesis)
                iter_trans = self.sample_transitions(iter_trans, size=self.sample_size)
                for transitions, hypothesis in zip(iter_trans, iter_hyp):
                    curr_evidence.append([MarkovChain.marginal_likelihood(csr_matrix(transitions), csr_matrix(hypothesis) * k) for k in self.hyptrails_ks])
                evidence.append(curr_evidence)
            evidence_mean = np.nanmean(evidence, axis=0)
            evidence_std = sem(np.array(evidence), axis=0)
            return evidence_mean, evidence_std
        else:
            for transitions, hypothesis in zip(self.transitions, self.hypotheses):
                evidence.append([MarkovChain.marginal_likelihood(csr_matrix(transitions), csr_matrix(hypothesis) * k) for k in self.hyptrails_ks])
            return np.array(evidence), np.zeros(np.array(evidence).shape)

    def calculate_js_divergence(self) -> list:
        divergence = []
        if self.is_sampling:
            for _ in tqdm(range(self.runs), desc="Collecting evidence. "):
                curr_divergence = []
                iter_trans, iter_hyp = [], []
                for transitions, hypothesis in zip(self.transitions, self.hypotheses):
                    curr_transitions, curr_hypothesis = self.sample_states(transitions=transitions, hypothesis=hypothesis, size=self.sample_size)
                    iter_trans.append(curr_transitions)
                    iter_hyp.append(curr_hypothesis)
                iter_trans = self.sample_transitions(iter_trans, size=self.sample_size, normalise=True)
                for i, (transitions, hypothesis) in enumerate(zip(iter_trans, iter_hyp)):
                    curr_divergence.append(np.nanmean([jensenshannon(x, y) for x, y in zip(transitions, hypothesis)]))
                divergence.append(curr_divergence)
            divergence = np.array(divergence).T.tolist()
        else:
            for transitions, hypothesis in zip(self.transitions, self.hypotheses):
                divergence.append(np.nanmean([jensenshannon(x, y) for x, y in zip(transitions, hypothesis)]))
        return divergence

    def get_hyptrails_evidence(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :returns tuple[evidence means and evidence std]
        """
        return self.hyptrails_evidence

    def get_js_divergence(self) -> list:
        return self.js_divergence

    def get_edges(self) -> list:
        return [x.sum() for x in self.transitions]

    def get_densities(self) -> list:
        return [(np.count_nonzero(m) / (m.shape[0] * m.shape[1])) for m in self.transitions]


def print_hyptrails(evidences: Tuple[np.ndarray, np.ndarray], x_ticks: list, k_values: list, save_path: str) -> None:
    import matplotlib.pyplot as plt
    # plt.style.use('seaborn')
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 18,
        "font.size": 18,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
    width = 345
    plt.rcParams.update(tex_fonts)
    fig_size = set_size(width) 
    plt.figure(figsize=fig_size)
    tick_positions = np.arange(len(k_values))
    colors = ["steelblue", "firebrick", "mediumseagreen"]
    for label_idx, (name, evidence_mean, evidence_std) in enumerate(zip(x_ticks, evidences[0], evidences[1])):
        plt.plot(tick_positions, evidence_mean, label=f"{name}", color=colors[label_idx])
        plot_min = np.subtract(evidence_mean, evidence_std)
        plot_max = np.add(evidence_mean, evidence_std)
        plt.fill_between(tick_positions, plot_max, plot_min, alpha=.1, color=colors[label_idx])
    plt.xticks(tick_positions, k_values)
    plt.xlabel("k-factor")
    plt.ylabel("Evidence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")

def print_divergence(divergence: list, x_ticks: list, save_path: str, is_sampled: bool) -> None:
    import matplotlib.pyplot as plt 
    # plt.style.use('seaborn')
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 18,
        "font.size": 18,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
    width = 345
    plt.rcParams.update(tex_fonts)
    fig_size = set_size(width)
    plt.figure(figsize=fig_size)
    if is_sampled:
        import seaborn as sns
        sns.boxplot(data=divergence)
        plt.xticks(range(len(x_ticks)), x_ticks)
        plt.ylim(0, 1)
    else:
        counter = np.arange(0.25, 1, 0.25)
        for idx, unsampled in zip(counter, divergence):
            plt.axhline(y=unsampled, xmin=idx-0.05, xmax=idx+0.05, c='b')
        plt.xticks(range(len(x_ticks) + 2), [""] + x_ticks + [""])
    plt.xlabel("Network size")
    plt.ylabel("Divergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")


LOAD = False
k_values = [1, 3, 5, 10, 100, 1000]
runs = 1000
different_node_settings = {
    'name': 'node',
    'sizes': [10, 100, 1000], 
    'transition_counts': [5, 50, 500], 
    'walk_lengths': [5, 10, 25], 
    'walks_per_node': 50, 
    'data_generation': "barabasi", 
    'state_sample_method': "snowball", 
    'mixtures': [0.5, 1.0 , 0.5], 
    'x_ticks': [10, 100, 1000], 
}

different_edge_settings = {
    'name': 'edge',
    'sizes': [100, 100, 100], 
    'transition_counts': [10, 10, 10], 
    'walk_lengths': [10, 20, 30], 
    'walks_per_node': 5, 
    'data_generation': "barabasi", 
    'state_sample_method': "snowball", 
    'mixtures': [0.5, 1.0 , 0.5], 
    'x_ticks': ["sparse", "mean", "dense"], 
}


different_distr_settings = {
    'name': 'distr',
    'sizes': [100, 100, 100], 
    'transition_counts': [10, 20, 30], 
    'walk_lengths': [10, 10, 10], 
    'walks_per_node': 10, 
    'data_generation': "barabasi", 
    'state_sample_method': "snowball", 
    'mixtures': [0.5, 1.0 , 0.5], 
    'x_ticks': ["sparse", "mean", "dense"], 
}


if __name__ == '__main__':
    settings = different_edge_settings
    if LOAD: 
        sampled_js_divergence = np.load(open(Path("images", f"{settings['name']}-sampled_js.npy"), 'rb')).tolist()
        sampled_hyptrails_evidence = np.load(open(Path("images", f"{settings['name']}-sampled_hyp.npy"), 'rb'))
        unsampled_js_divergence = np.load(open(Path("images", f"{settings['name']}-unsampled_js.npy"), 'rb'))
        unsampled_hyptrails_evidence = np.load(open(Path("images", f"{settings['name']}-unsampled_hyp.npy"), 'rb'))
        print(f"Sampled JS: {np.nanmean(np.array(sampled_js_divergence), axis=1)}")
        print(f"Unsampled JS: {unsampled_js_divergence}")
    else:
        sampled_dataset = SyntheticDataset(args={
            'data_generation': settings['data_generation'],
            'sizes': settings['sizes'],
            'transition_counts': settings['transition_counts'],
            'walk_lengths': settings['walk_lengths'],
            'walks_per_node': settings['walks_per_node'], 
            'state_sample_method': settings['state_sample_method'], 
            'mixtures': settings['mixtures'],  # how much hypothesis is in there
            'hyptrails_ks': k_values,
            'runs': runs,
            'sample_size': 10,
            'is_sampling': True
        })
        unsampled_dataset = SyntheticDataset(args={
            'data_generation': settings['data_generation'],
            'sizes': settings['sizes'],
            'transition_counts': settings['transition_counts'],
            'walk_lengths': settings['walk_lengths'],
            'walks_per_node': settings['walks_per_node'], 
            'state_sample_method': settings['state_sample_method'], 
            'mixtures': settings['mixtures'],  # how much hypothesis is in there
            'hyptrails_ks': k_values,
            'runs': runs,
            'sample_size': 10,
            'is_sampling': False
        })
        sampled_js_divergence = sampled_dataset.get_js_divergence()
        sampled_hyptrails_evidence = sampled_dataset.get_hyptrails_evidence()
        unsampled_js_divergence = unsampled_dataset.get_js_divergence()
        unsampled_hyptrails_evidence = unsampled_dataset.get_hyptrails_evidence()
        for name, value in [("sampled_js", sampled_js_divergence), ("sampled_hyp", sampled_hyptrails_evidence), ("unsampled_js", unsampled_js_divergence), ("unsampled_hyp", unsampled_hyptrails_evidence)]:
            with open(Path("data", "synthetic", f"{settings['name']}-{name}.npy"), 'wb') as f:
                np.save(f, np.array(value))
        print(f"Edges: {sampled_dataset.get_edges()}/{unsampled_dataset.get_edges()}")
        print(f"Densities: {sampled_dataset.get_densities()}/{unsampled_dataset.get_densities()}")
    print_divergence(divergence=sampled_js_divergence, x_ticks=settings['x_ticks'], save_path=f"data/images/{settings['name']}_js_sampled", is_sampled=True)
    print_divergence(divergence=unsampled_js_divergence,x_ticks=settings['x_ticks'], save_path=f"data/images/{settings['name']}_js_unsampled", is_sampled=False)
    print_hyptrails(evidences=sampled_hyptrails_evidence, x_ticks=settings['x_ticks'], k_values=k_values, save_path=f"data/images/{settings['name']}_hyp_sampled")
    print_hyptrails(evidences=unsampled_hyptrails_evidence, x_ticks=settings['x_ticks'], k_values=k_values, save_path=f"data/images/{settings['name']}_hyp_unsampled")
