import os
import json
import enum
import numpy as np
import matplotlib.pyplot as plt

class EvalDataset(str, enum.Enum):
    WIKISPEEDIA = "WikiSpeedia"
    FlICKR = "FlICKR PhotoTrails"
    BIBLIOGMETRIC = "Bibliographic"
    BIBLIOGMETRIC_COUNTRY = "Bibliographic Country"
    BIBLIOGMETRIC_AFFILIATION = "Bibliographic Affiliation"
 
    def __str__(self):
        return self.value

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

def rename(label_name: str, titles: list) -> str:
        for i in range(len(titles)):
            label_name = label_name.replace(f"-{i}", f"-{titles[i][:3]}")
        return label_name

def rename_plot_title(dataset_type: EvalDataset, plot_title: str) -> str:
    if dataset_type == EvalDataset.FlICKR:
        if "Center" in plot_title:
            plot_title = "Center"
    elif dataset_type == EvalDataset.BIBLIOGMETRIC_COUNTRY:
        if "Same" in plot_title:
            plot_title = "Same Country"
    elif dataset_type == EvalDataset.BIBLIOGMETRIC_AFFILIATION:
        if "Same" in plot_title:
            plot_title = "Same Affi"
    return plot_title


def interpolate_nan(input_list: np.ndarray) -> list:
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, x = nan_helper(input_list)
    input_list[nans]= np.interp(x(nans), x(~nans), input_list[~nans])
    return input_list.tolist()



def plot_violin_realworld(dataset_type: EvalDataset, evidences: dict, amount_datasets: int, titles: list, black_listed_hypotheses: list = (), save_path: str = None) -> None:
    # plt.style.use('seaborn')
    width = 345
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 16,
        "font.size": 16,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
    plt.rcParams.update(tex_fonts)
    fig_size = set_size(width)

    labels = {x: [] for x in range(amount_datasets)}
    data = {x: [] for x in range(amount_datasets)}
    black_listed_hypotheses.append('conf')
    if dataset_type == EvalDataset.BIBLIOGMETRIC_AFFILIATION:
        desired_order = ['diff_affiliation-0', 'same_affiliation-0', 'distance-0', 'diff_affiliation-1', 'same_affiliation-1', 'distance-1', 'diff_affiliation-2', 'same_affiliation-2', 'distance-2', 'diff_affiliation-3', 'same_affiliation-3', 'distance-3', 'diff_affiliation-4', 'same_affiliation-4', 'distance-4', 'diff_affiliation-5', 'same_affiliation-5', 'distance-5', 'config']
        evidences = {k: evidences[k] for k in desired_order}
    for name, evi in evidences.items():
        if name[:-2] not in black_listed_hypotheses:
            evi = np.array(evi)
            # mask = ~np.isnan(evi[:, 0])
            # curr_data = [d[m].tolist() for d, m in zip(evi[:, 0], mask)]
            plot_identifier = int(name[-1])
            labels[plot_identifier].append(name)
            # data[plot_identifier].append([a for b in curr_data for a in b])  # flatten
            data[plot_identifier].append(interpolate_nan(evi[:, 0]))
    all_labels = np.array(list(labels.values()))
    data = np.array(list(data.values()))

    print(f"{str(dataset_type)}: {[(label[0], value.mean()) for label, value in zip(labels.values(), data)]}")
    _, ax = plt.subplots(1, data.shape[1], figsize=fig_size, sharey=True)
    for i in range(data.shape[1]):
        curr_data = data[:, i, ].tolist()
        curr_labels = [rename(label_name=x, titles=titles) for x in all_labels[:, i].tolist()]
        plot_title = " ".join(curr_labels[0].split("-")[:-1]).title()
        curr_labels = [x.split("-")[-1] for x in curr_labels]  # [""] + 
        if data.shape[1] > 1:
            ax[i].boxplot(curr_data)
            ax[i].set_xticks(np.arange(1, len(curr_labels) + 1))
            ax[i].set_xticklabels(curr_labels, rotation=70)
            if dataset_type not in [EvalDataset.BIBLIOGMETRIC, EvalDataset.BIBLIOGMETRIC_COUNTRY, EvalDataset.BIBLIOGMETRIC_AFFILIATION]:
                ax[i].set_ylim(0.0, 0.85)
            ax[i].title.set_text(f"Hyp: {rename_plot_title(dataset_type=dataset_type, plot_title=plot_title)}")
            ax[0].set_ylabel("Divergence")
            ax[i].grid("major")
        else:
            ax.boxplot(curr_data)
            ax.set_xticks(np.arange(1, len(curr_labels) + 1))
            ax.set_xticklabels(curr_labels, rotation=70)
            if dataset_type not in [EvalDataset.BIBLIOGMETRIC, EvalDataset.BIBLIOGMETRIC_COUNTRY, EvalDataset.BIBLIOGMETRIC_AFFILIATION]:
                ax.set_ylim(0.0, 0.85)
            ax.title.set_text(f"Hyp: {rename_plot_title(dataset_type=dataset_type, plot_title=plot_title)}" )
            ax.set_ylabel("Divergence")
            ax.grid("major")
    plt.savefig(save_path, bbox_inches="tight", format='png')
    

# eval_datasets = [EvalDataset.WIKISPEEDIA, EvalDataset.FlICKR, EvalDataset.BIBLIOGMETRIC, EvalDataset.BIBLIOGMETRIC_COUNTRY, EvalDataset.BIBLIOGMETRIC_AFFILIATION]
eval_datasets = [EvalDataset.BIBLIOGMETRIC]
SAMPLE_PLOT = False


if __name__ == '__main__':
    for eval_dataset in eval_datasets:
        if eval_dataset == EvalDataset.WIKISPEEDIA:
            paths = [os.path.join("data", "evidences", "wikispeedia.json")]
            titles = ["out", "in"]
            black_listed_hypotheses = ['cosine']
            save_path = os.path.join("data", "images", "wikispeedia")
        elif eval_dataset == EvalDataset.FlICKR:
            paths = [os.path.join("data", "evidences", "flickr.json")]
            titles = ["vancouver", "washington", "la", "london", "nyc"]
            black_listed_hypotheses = ["center_distance", "distance"]
            save_path = os.path.join("data", "images", "flickr")
        elif eval_dataset == EvalDataset.BIBLIOGMETRIC:
            paths = [os.path.join("data", "evidences", "bibliometric.json")]
            titles = ["ss", "dm", "ai", "sp", "r", "hci"]
            black_listed_hypotheses = ["affi_graph", "cognitive_graph", "country_graph", "data", "hub", "cognitive", "affi"]
            save_path = os.path.join("data", "images", "bibliometric")
        elif eval_dataset == EvalDataset.BIBLIOGMETRIC_COUNTRY:
            paths = [os.path.join("data", "evidences", "bibliometric_country.json")]
            titles = ["ss", "dm", "ai", "sp", "r", "hci"]
            black_listed_hypotheses = ["diff_country", "same_country"]
            save_path = os.path.join("data", "images", "bibliometric_country")
        elif eval_dataset == EvalDataset.BIBLIOGMETRIC_AFFILIATION:
            paths = [os.path.join("data", "evidences", "bibliometric_affiliation.json")]
            titles = ["ss", "dm", "ai", "sp", "r", "hci"]
            black_listed_hypotheses = ["diff_affiliation", "same_affiliation"]
            save_path = os.path.join("data", "images", "bibliometric_affiliation")
        else:
            print("ERROR: No such data_generation. ")
            exit(1)
        for sampling_strategy in [('random', 'Random'), ("Snowball on Transition", "transitions"), ("Snowball on Hypothesis", "hypotheses")]:
            print("Evaluating ", eval_dataset.value, " with ", sampling_strategy, ". ")
            evidences = load_and_filter(paths=paths, filters={'sampling_strategies': sampling_strategy[0]})[0]
            plot_violin_realworld(dataset_type=eval_dataset, evidences=evidences, black_listed_hypotheses=black_listed_hypotheses, amount_datasets=len(titles),titles=titles, save_path=f"{save_path}_{sampling_strategy[1]}.png" if save_path else None)
            print(89 * "-")
            print()

