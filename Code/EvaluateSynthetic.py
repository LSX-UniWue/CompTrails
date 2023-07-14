import os
import json
import enum
import numpy as np
import matplotlib.pyplot as plt


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
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def load_and_filter(paths: list, filters: dict) -> list:
    evidences = []
    for path in paths:
        print("Loading file: ", path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                evidences.append(json.loads(line))
    for x, y in filters.items():
        evidences = [evi for evi in evidences if evi['config'][x] == y]
    return evidences


def my_plot(evidences: dict, configs: list, label_name: str, transition_weighted: bool, save_path: str, subplot_titles: list = None, perc: int = 100) -> None:
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
    plt.rcParams.update(tex_fonts)

    labels = []
    amount_hyps = round(len(evidences) / 2)
    for idx in range(2):  # amount of hypotheses
        curr_labels = []
        for config in configs:
            curr_label = round(config["transition_probas"][idx][label_name], 3)
            # curr_labels.append("")
            if curr_label in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
                curr_labels.append(round(config["transition_probas"][idx][label_name], 3))
            else:
                curr_labels.append("")
            labels.append(curr_labels)
    fig_size = (amount_hyps * 4, 4) 
    # fig_size = set_size(width, subplots=(1 if same_plot else amount_datasets, amount_hyps))
    _, ax = plt.subplots(1, amount_hyps, figsize=fig_size, sharey=True)
    colors = ["royalblue", "darkred"]
    for idx, (name, evi) in enumerate(evidences.items()):
        # maybe put all of one data_generation in the same plot?
        label_idx = idx % 2
        number_runs = evi.shape[1]
        only_evi = evi[:, :int(number_runs * perc / 100), 0]
        if transition_weighted:
            data = []
            for i in range(only_evi.shape[0]):  # usually 17
                curr = []
                for val in evi[i, :, [0, 1]].T:  # something like 100x2
                    if not np.isnan(val[0]):
                        curr.extend(np.repeat(val[0], val[1]))
                data.append(curr)
        else:
            mask = ~np.isnan(only_evi)
            data = [d[m].tolist() for d, m in zip(only_evi, mask)]
        # Filter data using np.isnan
        plot_identifier = int(idx / 2)
        error_plot_data = np.array([(np.nanmean(x), np.nanstd(x)) for x in data])
        error_plot_mean = error_plot_data[:, 0]
        error_plot_min = np.subtract(error_plot_data[:, 0], error_plot_data[:, 1])
        error_plot_max = np.add(error_plot_data[:, 0], error_plot_data[:, 1])
        positions = np.arange(0, 11)
        ax[plot_identifier].plot(positions, error_plot_mean, color=colors[label_idx], label=LEGEND[label_idx])
        ax[plot_identifier].fill_between(positions, error_plot_max, error_plot_min, alpha=.1, color=colors[label_idx])
        if subplot_titles:
            ax[plot_identifier].set_title(f"Hyp: {subplot_titles[plot_identifier]}")
        else:
            ax[plot_identifier].set_title(f"Hyp: {name.split('_')[0].split('-')[0]}")
        if idx == len(evidences) - 1:
            ax[plot_identifier].legend()
        ax[plot_identifier].set_xticks(np.arange(0, len(labels[label_idx])))
        ax[plot_identifier].set_xticklabels(["" for _ in range(len(labels[label_idx]))], rotation=70)
        ax[plot_identifier].set_xlabel(r'$\longleftarrow$ even / odd $\longrightarrow$' "\n" r'network')
        ax[plot_identifier].set_ylim(0.0, 0.85)
        ax[plot_identifier].grid(True)
    ax[0].set_ylabel(r"Divergence")
    plt.savefig(save_path, bbox_inches="tight", format='png')


class EvalDataset(str, enum.Enum):
    BARABASI = "Barabasi-Albert"
    BARABASI_DENSITY = "Barabasi-Albert Different Density"
    BARABASI_NOSAMPLE = "Barabasi-Albert No Sample"
    BARABASI_NOGRAPH = "Barabasi-Albert with graph structure"
    BARABASI_ABSTRACTION = "Barabasi-Albert Reduce"
    BARABASI_IMPORTANCE = "Barabasi-Albert Importance"
    BARABASI_IMPORTANCE_REDUCE = "Barabasi-Albert Importance Reduce"

    def __str__(self):
        return self.value


eval_datasets = [EvalDataset.BARABASI]
LEGEND = (200, 1000)


if __name__ == '__main__':
    for eval_dataset in eval_datasets:
        if eval_dataset == EvalDataset.BARABASI:
            paths = [os.path.join("data", "evidences", "barabasi_evidences_even.json")]
            hypotheses = ['even_graph-0', 'even_graph-1', 'odd_graph-0', 'odd_graph-1', 'uni-0', 'uni-1']
            label = 'even'
            subplot_titles = None
            save_path = os.path.join("data", "images", "barabasi_evidences_even")
        elif eval_dataset == EvalDataset.BARABASI_DENSITY:
            paths = [os.path.join("data", "evidences", "barabasi_evidences_even_diff_densities.json")]
            hypotheses = ['even_graph-0', 'even_graph-1', 'odd_graph-0', 'odd_graph-1', 'uni-0', 'uni-1']
            label = 'even'
            subplot_titles = None
            save_path = os.path.join("data", "images", "barabasi_evidences_density")
        elif eval_dataset == EvalDataset.BARABASI_NOSAMPLE:  
            paths = [os.path.join("data", "evidences", "barabasi_unsampled_evidences.json")]
            hypotheses = ['even_graph-0', 'even_graph-1', 'odd_graph-0', 'odd_graph-1', 'uni-0', 'uni-1']
            label = 'even'
            subplot_titles = None
            save_path = os.path.join("data", "images", "barabasi_unsampled_evidences")
        elif eval_dataset == EvalDataset.BARABASI_NOGRAPH:
            paths = [os.path.join("data", "evidences", "barabasi_evidences_even.json")]
            hypotheses = ['even-0', 'even-1', 'odd-0', 'odd-1', 'teleport-0', 'teleport-1']
            label = 'even'
            subplot_titles = ["Even-global", "Odd-global", "Uni-global"]
            save_path = os.path.join("data", "images", "barabasi_evidences_even_graph")
        elif eval_dataset == EvalDataset.BARABASI_ABSTRACTION:
            paths = [os.path.join("data", "evidences", "barabasi_evidences_even_reduced.json")]
            hypotheses = ['even-0', 'even-1', 'odd-0', 'odd-1', 'uni-0', 'uni-1']
            label = 'even'
            subplot_titles = None
            save_path = os.path.join("data", "images", "barabasi_evidences_even_reduced")
        elif eval_dataset == EvalDataset.BARABASI_IMPORTANCE:
            paths = [os.path.join("data", "evidences", "barabasi_evidences_importance.json")]
            hypotheses = ['only_least_important-0', 'only_least_important-1', 'only_most_important-0', 'only_most_important-1', 'uni-0', 'uni-1']
            label = 'only_most_important'
            subplot_titles = ['high', 'low', 'uni']
            save_path = os.path.join("data", "images", "barabasi_evidences_importance")
        elif eval_dataset == EvalDataset.BARABASI_IMPORTANCE_REDUCE:
            paths = [os.path.join("data", "evidences", "barabasi_evidences_importance_reduced.json")]
            hypotheses = ['importance-0-0', 'importance-0-1', 'importance-9-0', 'importance-9-1', 'uni-0', 'uni-1']
            label = 'only_most_important'
            subplot_titles = ["high", "low", "uni"]
            save_path = os.path.join("data", "images", "barabasi_evidences_importance_reduced")
        else:
            print("ERROR: No such data_generation. ")
            exit(1)
        for sampling_strategy in [('random', 'Random'), ("Snowball on Transition", "transitions"), ("Snowball on Hypothesis", "hypotheses")]:
            print(f"Evaluating {eval_dataset.value} with {sampling_strategy}. ")
            evidences = load_and_filter(paths=paths, filters={'sampling_strategies': sampling_strategy[0]})
            selected = {x: np.array([evi[x] for evi in evidences]) for x in hypotheses}
            configs = [evi['config'] for evi in evidences]
            # my_plot(evidences=selected, configs=configs, label_name=label, save_path=f"{save_path}_{sampling_strategy[1]}.png", subplot_titles=subplot_titles, transition_weighted=False)
            for i in [10, 30, 50, 70]:
                my_plot(evidences=selected, configs=configs, label_name=label, save_path=f"{save_path}_runs{i}_{sampling_strategy[1]}.png", subplot_titles=subplot_titles, transition_weighted=False, perc=i)
