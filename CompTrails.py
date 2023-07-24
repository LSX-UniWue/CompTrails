import os
import copy
import sys
import json
import time
from typing import Tuple
from tqdm import tqdm

from scipy.stats import entropy
from sklearn.preprocessing import normalize
from pytrails.hyptrails import MarkovChain

from Code import *
from Code.datasets.AbstractDatasets import load_real_world_dataset
from Code.datasets.BarabasiDatasets import *
from Code.datasets.WikispeediaDataset import WikipediaDataset
from Code.datasets.FlickrDataset import FlickrDataset


class CompTrails:
    def __init__(self, args: dict):
        self.save_dir = args['save_dir']
        self.save_file = args['save_file'] if 'save_file' in args else None
        self.sampling = args['sampling']
        self.sampling_strategies = args['sampling_strategies']
        self.fixed_transitions_sampling_count = args['fixed_transitions_sampling_count'] if 'fixed_transitions_sampling_count' in args else None
        self.sample_percentage = args['sample_percentage'] if 'sample_percentage' in args else 1.0
        self.number_samples = args['number_samples']
        self.eval_method = args["eval_method"]
        self.verbose = args['verbose']
        self.dataset_name = args['data_generation']
        self.overall_time = time.time()
        if args['data_generation'] == ComptrailsDataset.BARABASIALBERT:
            self.dataset = BarabasiAlbertDataset(args=args)
        elif args['data_generation'] == ComptrailsDataset.BIBLIOMETRIC:
            # self.dataset = BibliometricDataset(args=args['bib_dataset'])
            print("ERROR: Bibliometric dataset is only supported using loaded dataset. ")
            exit(1)
        elif args['data_generation'] == ComptrailsDataset.WIKISPEEDIA:
            self.dataset = WikipediaDataset(args=args['wikispeedia_dataset'])
        elif args['data_generation'] == ComptrailsDataset.FLICKR:
            self.dataset = FlickrDataset(args=args['flickr_dataset'])
        elif args['data_generation'] == ComptrailsDataset.LOADEDREALWORLD:
            self.dataset = load_real_world_dataset(args['loaded_dataset_path'])
        else:
            print("ERROR: Unknown data_generation: ", args['data_generation'])
            exit(1)
        self.transitions = self.dataset.get_transitions()
        if self.verbose == 2:
            for transition in self.transitions:
                print("\t\t", transition.shape)
        self.sizes = [x.shape[0] for x in self.transitions]
        if self.verbose == 2:
            print("\t\tCalculating transition density. ")
        self.transition_density = [self.get_density(tran) for tran in self.transitions]
        self.amount_transitions = [self.transitions[i].sum() for i in range(len(self.transitions))]
        if self.verbose == 2:
            print(f"\t\tNodes: {self.sizes} with densities {self.transition_density} and amount of transitions {self.amount_transitions} ")
        self.hypotheses = self.dataset.get_hypotheses()
        evidences = []
        for sampling_strategy in self.sampling_strategies:
            if self.verbose:
                print(f"Running {sampling_strategy} as sampling strategy. ")
            curr_evidences = {}
            for i, dataset in enumerate(self.hypotheses):
                for name in dataset.keys():
                    curr_evidences[f"{name}-{i}"] = np.zeros((self.number_samples, 3))
            for idy in tqdm(range(self.number_samples), desc="Running evaluation"):
                results = self.evaluate(sampling_strategy)
                for result in results:
                    curr_evidences[result['id']][idy][0] = result['divergence']
                    curr_evidences[result['id']][idy][1] = result['transition_sum']
                    curr_evidences[result['id']][idy][2] = result['density']
            evidences.append(curr_evidences)
        if self.verbose:
            print("Saving ... ")
        for sampling_strategy, curr_evidences in zip(self.sampling_strategies, evidences):
            for k, v in curr_evidences.items():
                curr_evidences[k] = v.tolist()
            save_config = {"data_generation": self.dataset_name.value,
                            "sampling": self.sampling,
                            "eval_method": self.eval_method.value,
                            "sizes": self.sizes,
                            "sampling_strategies": sampling_strategy.value,
                            "number_samples": self.number_samples,
                            "sample_percentage": self.sample_percentage,
                            "max_transition_count": args['max_transition_count'],
                            "transition_probas": args['transition_probas'],
                            "transitions": np.array(self.amount_transitions).tolist(),
                            "density": np.array(self.transition_density).tolist(),
                            }
            for k, v in save_config.items():
                print(f"Key: {k} - Value: {v}")
            curr_evidences['config'] = save_config
            with open(os.path.join(self.save_dir, self.save_file), 'a', encoding='utf-8') as f:
                json.dump(curr_evidences, f)
                f.write("\n")
        print(f"Finished. Overall time is {round(time.time() - self.overall_time)} seconds. ")

    def evaluate(self, sampling_strategy: StateSamplingStrategy) -> list:
        def kl_div(x: csr_matrix, y: csr_matrix):
            return np.nanmean(np.multiply(x.A, np.log(x / y)))

        transitions = self.transitions.copy()
        hypotheses = self.hypotheses.copy()
        result = []

        if self.sampling:
            sampled_transitions, sampled_hypotheses = self.coherent_sampling(transitions=transitions, hypothesis=hypotheses, sampling_strategy=sampling_strategy)
        else:
            sampled_transitions = [[copy.deepcopy(transitions[idx]) for _ in range(len(hypotheses[0]))] for idx in range(len(transitions))]
            sampled_hypotheses = copy.deepcopy(hypotheses)

        for i in range(len(sampled_transitions)):
            for hyp_idx, (name, sampled_hypothesis) in enumerate(sampled_hypotheses[i].items()):
                if self.eval_method == EvaluationMetric.JENSONSHANNON:
                    def sparse_js(p, q):
                        # handmade JS Divergence for sparse things
                        m = (p + q) / 2.0
                        js = kl_div(p, m) + kl_div(q, m)
                        return np.sqrt(js / 2.0)

                    divergence = [sparse_js(normalize(sampled_transitions[i][hyp_idx][j], norm='l1', axis=1), sampled_hypothesis[j]) for j in range(sampled_hypothesis.shape[0])]
                elif self.eval_method == EvaluationMetric.HYPTRAILS:
                    divergence = [MarkovChain.marginal_likelihood(sampled_transitions[i][hyp_idx][j], sampled_hypothesis[j] * self.hyptrails_k) for j in range(sampled_hypothesis.shape[0])]
                else:
                    print("ERROR: Cannot find eval method: ", self.eval_method)
                    sys.exit(1)
                result.append({'id': name + "-" + str(i),
                               'divergence': np.nanmean(divergence),
                               'transition_sum': sampled_transitions[i][hyp_idx].sum(),
                               'density': self.get_density(sampled_transitions[i][hyp_idx])})
        return result

    @staticmethod
    def sample_states(adjacency: dict, sample_size: int, sampling_strategy: StateSamplingStrategy) -> list:
        if sampling_strategy == StateSamplingStrategy.RANDOM:
            return list(np.random.randint(adjacency['transition'].shape[0], size=sample_size))
        if sampling_strategy == StateSamplingStrategy.SNOWBALL_TRANSITION or sampling_strategy == StateSamplingStrategy.SNOWBALL_HYPOTHESIS:
            if sampling_strategy == StateSamplingStrategy.SNOWBALL_TRANSITION:
                transition_matrix = adjacency['transition']
            else:  # if sampling_strategy == StateSamplingStrategy.SNOWBALL_HYPOTHESIS:
                transition_matrix = adjacency['hypothesis']
            sampled_states = [np.random.randint(transition_matrix.shape[0])]  # starting point
            new_seeds = set(sampled_states)
            hops = 0
            while len(sampled_states) < sample_size:
                hops += 1
                neighbors = set()
                for seed in new_seeds:
                    neighbors.update(np.nonzero(transition_matrix[seed])[-1])
                new_seeds = neighbors - new_seeds
                nodes_left = sample_size - len(sampled_states)
                if len(new_seeds) >= nodes_left:
                    sampled_states.extend(np.random.choice(list(new_seeds), nodes_left))
                elif len(new_seeds) == 0 or hops == 2:  # restart
                    new_seeds = {np.random.randint(transition_matrix.shape[0])}
                    sampled_states.extend(new_seeds)
                    hops = 0
                else:
                    sampled_states.extend(neighbors)
                    new_seeds = neighbors
            return list(sampled_states[:sample_size])

    def coherent_sampling(self, transitions, hypothesis, sampling_strategy: StateSamplingStrategy) -> Tuple[list, list]:
        """
        :param transitions:
        :param hypothesis:
        :param sampling_strategy:
        :return: Transitions are not normalized
        """
        # deepcopy list of dicts
        curr_transitions = transitions.copy()
        curr_hypotheses = [copy.deepcopy(x) for x in hypothesis]
        amount_matrices = len(curr_transitions)
        for check_i in range(amount_matrices):
            for hyp in curr_hypotheses[check_i].keys():
                if curr_transitions[check_i].shape[0] != curr_hypotheses[check_i][hyp].shape[0]:
                    print(f'ERROR: Not correct shapes for: {check_i} - {hyp}:{curr_transitions[check_i].shape[0]}/{curr_hypotheses[check_i][hyp].shape[0]}')
                    exit(1)
        min_size = min(t.shape[0] for t in curr_transitions)
        sample_size = max(5, int(min_size * self.sample_percentage))
        if self.verbose == 2:
            print(f"Sampling to {sample_size} transitions. ")
        sampled_state_transitions, sampled_state_hypotheses = \
            [[] for _ in range(amount_matrices)], [{} for _ in range(amount_matrices)]
        # for each matrix
        for matrix_i in range(amount_matrices):
            # For each hypothesis
            for hyp_name in curr_hypotheses[matrix_i].keys():
                # randomly sample min_size amount of states with "zurücklegen"
                rand_idx = self.sample_states(adjacency={'transition': curr_transitions[matrix_i], 'hypothesis': curr_hypotheses[matrix_i][hyp_name]}, sample_size=sample_size, sampling_strategy=sampling_strategy)
                sample_transitions = curr_transitions[matrix_i][:, rand_idx][rand_idx, :]
                curr_hypotheses[matrix_i][hyp_name] = curr_hypotheses[matrix_i][hyp_name][:, rand_idx][rand_idx, :]
                # Sort transitions by amount of transitions
                sort_idx = np.array(sample_transitions.sum(axis=1).flatten()).argsort()[0][::-1]
                sample_transitions = sample_transitions[sort_idx, :][:, sort_idx]
                sampled_state_transitions[matrix_i].append(sample_transitions)
                curr_hypotheses[matrix_i][hyp_name] = curr_hypotheses[matrix_i][hyp_name][sort_idx, :][:, sort_idx]
                if any([x.sum() == 0 for x in curr_hypotheses[matrix_i][hyp_name]]):
                    curr_hypotheses[matrix_i][hyp_name] = save_normalize(curr_hypotheses[matrix_i][hyp_name])
                else:
                    # Remove Nan values
                    curr_hypotheses[matrix_i][hyp_name][curr_hypotheses[matrix_i][hyp_name] != curr_hypotheses[matrix_i][hyp_name]] = 0
                    curr_hypotheses[matrix_i][hyp_name] = normalize(curr_hypotheses[matrix_i][hyp_name], norm='l1', axis=1)
                sampled_state_hypotheses[matrix_i][hyp_name] = curr_hypotheses[matrix_i][hyp_name]
        sampled_transition_transitions = [[] for _ in range(amount_matrices)]
        for matrix_number in range(len(sampled_state_transitions[0])):
            current = [[] for _ in range(amount_matrices)]
            for row in range(sample_size):
                if self.fixed_transitions_sampling_count:
                    amount_sampled_transitions = self.fixed_transitions_sampling_count
                else:
                    amount_sampled_transitions = int(min(m[matrix_number][row].sum() for m in sampled_state_transitions))
                for matrix_idx in range(amount_matrices):
                    if amount_sampled_transitions > 0 and sampled_state_transitions[matrix_idx][matrix_number][row].sum() > 0:
                        current_row = sampled_state_transitions[matrix_idx][matrix_number][row].A
                        prob_distr = current_row / current_row.sum()
                        samples = np.random.choice(sample_size, size=amount_sampled_transitions, p=prob_distr[0])
                        reduce_by_key = np.array([len(np.where(samples == x)[0]) for x in range(sample_size)])
                        current[matrix_idx].append(reduce_by_key)
                    else:
                        current[matrix_idx].append(np.zeros(sample_size))
            for i, x in enumerate(current):
                sampled_transition_transitions[i].append(csr_matrix(x))
        return sampled_transition_transitions, sampled_state_hypotheses

    @staticmethod
    def get_density(m: csr_matrix) -> float:
        if isinstance(m, csr_matrix):
            return m.count_nonzero() / (m.shape[0] * m.shape[1])
        else:
            return np.count_nonzero(m) / (m.shape[0] * m.shape[1])

    def plots(self, evidences: dict, plot_type: PlotType, title: str = None) -> None:
        import matplotlib.pyplot as plt
        data = np.array([v for _, v in sorted(evidences.items())])[:, :, 0]
        for row_idx in range(len(data)):
            for cell_idx in range(len(data[row_idx])):
                if np.isnan(data[row_idx, cell_idx]):
                    data[row_idx, cell_idx] = np.nanmean(data[row_idx])
        density = np.nanmean(np.array([v for _, v in sorted(evidences.items())])[:, :, 2], axis=1)
        entropy = np.nanmean(np.array([v for _, v in sorted(evidences.items())])[:, :, 3], axis=1)
        labels = [k for k, _ in sorted(evidences.items())]
        fig, ax = plt.subplots()
        plt.title(title)
        if plot_type == PlotType.BOXPLOTS:
            ax.boxplot(data.T)
            labels = [a + " (" + str(round(b, 2)) + "/" + str(round(c, 2)) + ")" for a, b, c in
                      zip(labels, density, entropy)]
        elif plot_type == PlotType.VIOLINPLOT:
            ax.violinplot(data.tolist(), showmeans=True)
            labels = [a + " (" + str(round(b, 2)) + "/" + str(round(c, 2)) + ")" for a, b, c in
                      zip(labels, density, entropy)]
        else:  # will be PlotType.STDERROR
            ax.errorbar(np.arange(data.shape[0]), data.mean(axis=1), yerr=data.std(axis=1),
                        linestyle="None", fmt="-o", capsize=5)
            labels = ["This is a dummy, because set xticklabel is wierd"] + \
                     [a + " (" + str(round(b, 2)) + "/" + str(round(c, 2)) + ")" for a, b, c in
                      zip(labels, density, entropy)]
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(0, len(labels)) + 0.5)
        ax.set_xticklabels(labels, rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def bar_plot(self, matrix: np.ndarray, title: str = None, color: str = "r"):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        num_positions = len(matrix.reshape(-1, 1))
        xpos = np.array([[x for x in range(len(matrix))] for _ in range(len(matrix))]).flatten()
        ypos = np.array([[x for _ in range(len(matrix))] for x in range(len(matrix))]).flatten()
        zpos = np.zeros(num_positions)
        dx = np.ones(num_positions) * 0.5
        dy = np.ones(num_positions) * 0.5
        dz = matrix.reshape(1, -1)[0]
        # ax1.set_zlim(0.0, 1.0)
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color)
        if title:
            m_entropy = np.nansum([entropy(row) for row in matrix])
            plt.title(title + " Entropy: " + str(m_entropy))
        plt.show()


if __name__ == '__main__':
    STEPS = np.arange(0.0, 1.01, 0.1)
    dataset = sys.argv[1]
    print(f"Running dataset {dataset} ;) ")
    if dataset == "barabasi":  # Does random sampling, snowball on transition and snowball on hypothesis sampling, and no gsf as well
        configs = [{"data_generation": ComptrailsDataset.BARABASIALBERT, 'number_samples': 100, 'sample_percentage': 0.1, 'max_transition_count': [5, 25], 'amount_walks': [5, 25], 'max_walk_len': [5, 5], 'barabasi_transition_probas': [{'even': round(1.0 - i, 2), 'odd': round(i, 2)}, {'even': round(1.0 - i, 2), 'odd': round(i, 2)}], 'save_file': f'barabasi_evidences_even.json'} for i in STEPS]
    elif dataset == "barabasi-diff-densities":  
        configs = [{"data_generation": ComptrailsDataset.BARABASIALBERT, 'number_samples': 100, 'sample_percentage': 0.1, 'max_transition_count': [5, 5], 'amount_walks': [5, 5], 'max_walk_len': [5, 5], 'barabasi_transition_probas': [{'even': round(1.0 - i, 2), 'odd': round(i, 2)}, {'even': round(1.0 - i, 2), 'odd': round(i, 2)}], 'save_file': f'barabasi_evidences_even_diff_densities.json'} for i in STEPS]
    elif dataset == "barabasi-no-sample":  
        configs = [{"data_generation": ComptrailsDataset.BARABASIALBERT, 'sampling': False, 'number_samples': 1, 'sample_percentage': 0.1, 'max_transition_count': [5, 25], 'amount_walks': [5, 15], 'max_walk_len': [5, 5], 'barabasi_transition_probas': [{'even': round(1.0 - i, 2), 'odd': round(i, 2)}, {'even': round(1.0 - i, 2), 'odd': round(i, 2)}], 'save_file': 'barabasi_unsampled_evidences.json'} for i in STEPS]
    elif dataset == "barabasi-binary":  
        configs = [{"data_generation": ComptrailsDataset.REDUCEDBARABASIALBERT, 'sample_percentage': 1.0, 'max_transition_count': [2, 2], 'amount_walks': [1, 1], 'max_walk_len': [4, 4], 'barabasi_transition_probas': [{'even': round(1.0 - i, 2), 'odd': round(i, 2)}, {'even': round(1.0 - i, 2), 'odd': round(i, 2)}], 'save_file': 'barabasi_abstracted_evidences.json'} for i in STEPS]
    elif dataset == "barabasi-classes":  
        configs = [{"data_generation": ComptrailsDataset.REDUCEDBARABASIALBERT, 'max_importance': 10, 'use_importance': True, 'sample_percentage': 1.0, 'max_transition_count': [2, 2], 'amount_walks': [1, 1], 'max_walk_len': [4, 4], 'barabasi_transition_probas': [{'only_most_important': round(1.0 - i, 2), 'only_least_important': round(i, 2)}, {'only_most_important': round(1.0 - i, 2), 'only_least_important': round(i, 2)}], 'save_file': 'barabasi_abstracted_evidences.json'} for i in STEPS]
    elif dataset == "wikispeedia":
        configs = [{"data_generation": ComptrailsDataset.WIKISPEEDIA, 'sample_percentage': 0.1, "number_samples": 100, "save_file": "wikispeedia.json"}]
    elif dataset == "wikispeedia-loaded":
        configs = [{'data_generation': ComptrailsDataset.LOADEDREALWORLD, 'save_file': 'wikispeedia.json', 'sample_percentage': 0.3, "number_samples": 100, 'loaded_dataset_path': os.path.join('data', 'wikispeedia', f'wiki_matrices')}]
    elif dataset == "flickr":
        configs = [{"data_generation": ComptrailsDataset.FLICKR, 'sample_percentage': 1.0, "save_file": "flickr.json"}]
    elif dataset == "loaded-bib":
        configs = [{"data_generation": ComptrailsDataset.LOADEDREALWORLD, 'number_samples': 100, 'sample_percentage': 0.1, "save_file": "bibliometric.json", 'loaded_dataset_path': os.path.join('data', 'bibliometric', 'bibliometric_matrices')}]
    elif dataset == "loaded-bib-country":
        configs = [{"data_generation": ComptrailsDataset.LOADEDREALWORLD, 'number_samples': 100, 'sampling': False,  'sample_percentage': 0.1, "save_file": "bibliometric_country.json", 'loaded_dataset_path': os.path.join('data', 'bibliometric', 'bibliometric_matrices_country')}]
    elif dataset == "loaded-bib-affiliation":
        configs = [{"data_generation": ComptrailsDataset.LOADEDREALWORLD, 'number_samples': 1000, 'sample_percentage': 0.01, "save_file": "bibliometric_affiliation.json", 'loaded_dataset_path': os.path.join('data', 'bibliometric', 'bibliometric_matrices_affiliation')}]
    else:
        print(f"ERROR: Cannot create conig for {dataset}")
        exit(1)
    for i, config in enumerate(configs):
        print(f"Lets goo! Dataset: {config['data_generation'].value}, currently running config {i + 1}/{len(configs)}.")
        CompTrails(args={
            'sampling': config["sampling"] if "sampling" in config else True,
            'save_dir': os.path.join("data", "evidences"),
            'save_file': config["save_file"],
            'sizes': config["sizes"] if "sizes" in config else [200, 1000],
            'eval_method': config['eval_method'] if 'eval_method' in config else EvaluationMetric.JENSONSHANNON,
            'sample_percentage': config['sample_percentage'] if 'sample_percentage' in config else 1.0,
            'number_samples': config['number_samples'] if 'number_samples' in config else 1000,  # number of samples taken
            'sampling_strategies': [StateSamplingStrategy.SNOWBALL_HYPOTHESIS, StateSamplingStrategy.SNOWBALL_TRANSITION, StateSamplingStrategy.RANDOM],
            'graph_type': config['graph_type'] if 'graph_type' in config else 'barabasi',
            'data_generation': config['data_generation'],
            'loaded_dataset_path': config['loaded_dataset_path'] if 'loaded_dataset_path' in config else None,
            'max_transition_count': config['max_transition_count'] if 'max_transition_count' in config else [5, 5],  # If floats are given, they will be treated as percentages
            'transition_probas': config["barabasi_transition_probas"] if "barabasi_transition_probas" in config else [{'even': 1.0, 'odd': 0.0}, {'even': 0.0, 'odd': 1.0}],  # Barabasi Albert Dataset
            'amount_walks': config['amount_walks'] if 'amount_walks' in config else [10, 10],  # Barabasi Albert Dataset
            'max_walk_len': config['max_walk_len'] if 'max_walk_len' in config else [10, 10],  # Barabasi Albert Dataset
            'max_importance': config['max_importance'] if 'max_importance' in config else 10,  # Barabasi Albert Dataset
            'use_importance': config['use_importance'] if 'use_importance' in config else False,  # Barabasi Albert Dataset
            'wikispeedia_dataset': {
                'basedir': os.path.join("data", "wikispeedia"),
                'type': config['type'] if 'type' in config else 'default',
                'bins': config['bins'] if 'bins' in config else 10, 
                'steps': None,
                'test': False, 
                'sequence_path': os.path.join("wikispeedia_paths-and-graph", "paths_finished.tsv"),
                'article_path': os.path.join("wikispeedia_paths-and-graph", "articles.tsv"),
                'matrix_path': os.path.join("wikispeedia_paths-and-graph", "shortest-path-distance-matrix.txt"),
                'link_path': os.path.join("wikispeedia_paths-and-graph", "links.tsv"),
                'verbose': 1,
            },
            'flickr_dataset': {
                'basedir': os.path.join("data", "flickr-data"),
                'files': ["vancouver", "washington", "la", "london", "nyc"],
                'binarise': True if "flickr_binary" in config else False, 
                'verbose': 1
            },
            'bib_dataset': {
                'basedir': os.path.join("data", 'bibliometric'), 
                'datasets': ["ss", "dm", "ai", "sp", "r", "hci"],
                'type': config['type'] if 'type' in config else "default", 
                'k_sims': 100, 
                'debug': False, 
                'verbose': 1 
            },
            'verbose': 1
        })
