import os
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod

from pytrails.hyptrails import MarkovChain


def row_wise_normalize(matrix: np.array) -> np.array:
    for matrix_idx in range(matrix.shape[0]):
        if matrix[matrix_idx].sum() > 0:
            matrix[matrix_idx] = matrix[matrix_idx] / matrix[matrix_idx].sum()
    return csr_matrix(matrix)


def evaluate_dataset(dataset) -> None:
    import matplotlib.pyplot as plt
    max_k = 1000
    k_factors = np.arange(0, max_k + 1, max_k / 20).astype(int)

    def plot(evidences: dict) -> None:
        plt.figure()
        tick_positions = np.arange(len(k_factors))
        for name, evidence in evidences.items():
            plt.plot(tick_positions, evidence, marker=None, label=str(name))
        plt.xlabel("k-factor")
        plt.ylabel("Evidence")
        plt.legend()
        plt.xticks(tick_positions, k_factors, rotation=45)
        plt.tight_layout()
        plt.show()

    transitions = dataset.get_transitions()
    for transition_matrix in transitions:
        print(f"States: {transition_matrix.shape[0]} - edges: {transition_matrix.sum()} - sparsity: {1 - transition_matrix.nnz / (transition_matrix.shape[0] * transition_matrix.shape[1])}")
    hypotheses = dataset.get_hypotheses()
    evidences = {}
    for idx in range(len(transitions)):
        for hyp_name, matrix in hypotheses[idx].items():
            evidences[hyp_name] = [MarkovChain.marginal_likelihood(transitions[idx], matrix * c) for c in k_factors]
        plot(evidences=evidences)


def save_transitions_and_hypothesis(dataset, path: str, start_idx: int = None) -> None:
    transitions = dataset.get_transitions()
    hypotheses = dataset.get_hypotheses()
    for idx in range(len(transitions)):
        if start_idx:
            save_idx = idx + start_idx
        else:
            save_idx = idx
        scipy.sparse.save_npz(os.path.join(path, f"{save_idx}-transitions.npz"), transitions[idx])
        for name, matrix in hypotheses[idx].items():
            scipy.sparse.save_npz(os.path.join(path, f"{save_idx}-{name}.npz"), matrix)
    print("Transitions and hypotheses saved. ")


def load_real_world_dataset(dir_path: str):
    files = sorted(os.listdir(dir_path))
    amount_datasets = len(set([int([x for x in file if x.isdigit()][0]) for file in files]))
    transitions = []
    hypotheses = [{} for _ in range(amount_datasets)]
    for file in files:
        if ".npz" in file:
            matrix = scipy.sparse.load_npz(os.path.join(dir_path, file))
            idx = int(file.split("-")[0])
            name = file[:-4].split("-")[1]
            if name == "transitions":
                # check order
                transitions.append(matrix)
            else:
                hypotheses[idx][name] = matrix
    dataset = LoadedRealWorldDataset(args={'basedir': dir_path, 'verbose': 0})
    dataset.set_transitions(transitions=transitions)
    dataset.set_hypotheses(hypotheses=hypotheses)
    print("Finished loading the data set. ")
    return dataset


class SyntheticDataset(ABC):
    def __init__(self, args: dict):
        self.sizes = args['sizes']
        self.max_transition_count = args['max_transition_count']
        self.verbose = args['verbose']

    @abstractmethod
    def get_transitions(self) -> list:
        pass

    @abstractmethod
    def get_hypotheses(self) -> list:
        pass


class ReadWorldDataset(ABC):
    def __init__(self, args: dict):
        self.basedir = args['basedir']
        self.verbose = args['verbose']

    @abstractmethod
    def get_transitions(self) -> list:
        pass

    @abstractmethod
    def get_hypotheses(self) -> list:
        pass


class LoadedRealWorldDataset(ReadWorldDataset):
    def __init__(self, args: dict):
        super(LoadedRealWorldDataset, self).__init__(args=args)
        self.transitions = []
        self.hypotheses = []

    def get_transitions(self) -> list:
        return self.transitions

    def get_hypotheses(self) -> list:
        return self.hypotheses

    def set_transitions(self, transitions: list) -> None:
        self.transitions = transitions

    def set_hypotheses(self, hypotheses: list) -> None:
        self.hypotheses = hypotheses


if __name__ == '__main__':
    for dataset in [os.path.join("clickstream", "wiki_matrices"),
                    os.path.join("bibliometric", "bibliometric_matrices"),
                    os.path.join("higgs_twitter", "twitter_matrices")]:
        print(f"Doing {dataset}")
        dataset = load_real_world_dataset(os.path.join("data", dataset))
        evaluate_dataset(dataset=dataset)
