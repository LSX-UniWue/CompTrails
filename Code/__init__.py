import enum
import numpy as np


class ComptrailsDataset(str, enum.Enum):
    SYNTHETIC = "synthetic"
    BARABASIALBERT = "Barabasi-Albert Graph"
    REDUCEDBARABASIALBERT = "Barabasi-Albert Graph (reduced)"
    BIBLIOMETRIC = "Bibliometric"
    WIKISPEEDIA = "WikiSpeedia"
    FLICKR = "Flickr Phototrails"
    LOADEDREALWORLD = "Loaded Real World Dataset"

    def __str__(self):
        return self.value


class EvaluationMetric(str, enum.Enum):
    JENSONSHANNON = "Jensen Shannon"
    HYPTRAILS = "HypTrails"

    def __str__(self):
        return self.value


class PlotType(str, enum.Enum):
    BOXPLOTS = "Box Plot"
    STDERROR = "Std-Error Plot"
    VIOLINPLOT = "Violin Plot"


class StateSamplingStrategy(str, enum.Enum):
    RANDOM = "random"
    SNOWBALL_TRANSITION = "Snowball on Transition"
    SNOWBALL_HYPOTHESIS = "Snowball on Hypothesis"

    def __str__(self):
        return self.value


class TransitionSamplingStrategy(str, enum.Enum):
    NONE = "None"
    DISTRIBUTIONBASED = "Distribution-based"
    CONNECTIONBASED = "Connection-based"


def save_normalize(matrix: np.array) -> np.array:
    matrix_copy = matrix.copy()
    for row_idx in range(matrix_copy.shape[0]):
        if matrix_copy[row_idx].sum() > 0:
            matrix_copy[row_idx] = matrix_copy[row_idx] / matrix_copy[row_idx].sum()
    return matrix_copy
