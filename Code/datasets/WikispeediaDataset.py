import csv
import urllib
from typing import Tuple
from pathlib import Path
from scipy.sparse import dok_matrix
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import binarize, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from Code.datasets.AbstractDatasets import *


class WikipediaDataset(ReadWorldDataset):
    def __init__(self, args: dict):
        super().__init__(args)
        self.steps = args['steps'] if 'steps' in args else None
        self.type = args['type'] if 'type' in args else "default"
        self.bins = args['bins']
        self.sequences = self.load_sequences(Path(self.basedir, args['sequence_path']), test=args['test'])
        self.article_map = self.load_article_map(Path(self.basedir, args['article_path']))
        print("Indexing sequences ...")
        self.index_sequences = [[self.article_map[article] for article in sequence] for sequence in self.sequences]
        self.shortest_path_matrix = self.load_shortest_path_matrix(Path(self.basedir, args['matrix_path']))
        print("Filtering sequences with optimal length 3 ...")
        self.filtered_index_sequences = [s for s in self.index_sequences if self.shortest_path_matrix[s[0]][s[-1]] == 3]
        self.states = max([a for b in self.filtered_index_sequences for a in b]) + 1
        print(f"Filtered sequences: {len(self.filtered_index_sequences)}")
        print("Filtering sequences with 3-8 clicks ...")
        self.filtered_index_sequences = [s for s in self.filtered_index_sequences if 4 <= len(s) <= 9]
        print(f"Filtered sequences: {len(self.filtered_index_sequences)}")
        self.adjacency_matrix, self.degree, self.mean_degree = self.load_links(Path(self.basedir, args['link_path']))
        self.transitions, self.degree_transitions = self.create_transitions()
        if self.type == "degree":
            self.hypotheses = [{"degree": csr_matrix(np.triu(np.ones((self.bins, self.bins))) - np.diag(np.ones(self.bins)))}, {"degree": csr_matrix(np.triu(np.ones((self.bins, self.bins))) - np.diag(np.ones(self.bins)))}]
        else:
            self.cos_similarity = self.calculate_cosine()
            self.hypotheses = self.create_hypotheses()

    @staticmethod
    def load_sequences(sequence_path: str, test: bool = False) -> list:
        print("Loading sequences ...")
        sequences = []
        rows = (row for row in open(sequence_path) if not row.startswith('#'))
        for line in csv.reader(rows, delimiter='\t'):
            if len(line) == 0:
                continue
            seq = line[3].split(";")
            # for simplicity, let us remove back clicks
            seq = [urllib.parse.unquote(x) for x in seq if x != "<"]
            sequences.append(seq)
        print("Number of sequences:", len(sequences))
        if test:
            sequences = sequences[:2000]
        return sequences

    @staticmethod
    def load_article_map(article_path: str) -> dict:
        print("Article map ...")
        article_map = {}
        with open(article_path) as file:
            index = 0
            for line in file:
                stripped = line.strip()
                if stripped.startswith('#') or len(stripped) == 0:
                    continue
                article = urllib.parse.unquote(stripped)
                article_map[article] = index
                index += 1
        print("Number of articles:", len(article_map))
        return article_map

    @staticmethod
    def load_shortest_path_matrix(matrix_path: str) -> list:
        print("Loading shortest path matrix ...")
        shortest_path_matrix = []
        with open(matrix_path) as file:
            index = 0
            for line in file:
                stripped = line.strip()
                if stripped.startswith('#') or len(stripped) == 0:
                    continue

                def convert(x):
                    if x == "_":
                        return -1
                    else:
                        return int(x)

                shortest_path_matrix.append([convert(x) for x in stripped])
                index += 1
        return shortest_path_matrix

    def load_links(self, link_path: str) -> Tuple[dok_matrix, dict]:
        print("Loading links ...")
        inlinks = defaultdict(list)
        outlinks = defaultdict(list)
        adjacency_matrix = dok_matrix((len(self.article_map), len(self.article_map)))
        with open(link_path) as file:
            for line in file:
                if line.startswith('#') or len(line.strip()) == 0:
                    continue
                link = [self.article_map[urllib.parse.unquote(x.strip())] for x in line.split("\t")]
                outlinks[link[0]].append(link[1])
                inlinks[link[1]].append(link[0])
                adjacency_matrix[link[0], link[1]] = 1

        print("Calculating degrees ...")
        degree = dict([(i, len(indeg) + len(outlinks[i])) for i, indeg in inlinks.items()])
        mean_degree = int(np.mean(list(degree.values())))
        return adjacency_matrix, degree, mean_degree

    def calculate_cosine(self) -> csr_matrix:
        print("Calculating cosine similarity ...")

        article_text = np.empty((len(self.article_map),), dtype=object)

        for article, index in self.article_map.items():
            article_text[index] = open(Path(self.basedir, "plaintext_articles", urllib.parse.quote(article) + ".txt")).read()

        vect = TfidfVectorizer(max_df=0.8, sublinear_tf=True)
        X = vect.fit_transform(article_text)
        return X * X.T

    def create_hypotheses(self) -> list:
        single_hypotheses = {}
        print("Deriving hypotheses ...")

        adjacency_mask = self.adjacency_matrix.copy()
        adjacency_mask[adjacency_mask > 0] = 1
        hyp_cos = self.cos_similarity.multiply(adjacency_mask)
        single_hypotheses['cosine'] = csr_matrix(normalize(hyp_cos, norm='l1', axis=1))

        hyp_deg = np.zeros((len(self.article_map), len(self.article_map)))
        for k, v in tqdm(self.degree.items(), desc="Creating degree hypothesis"):
            curr = np.zeros(len(self.article_map))
            for k_o, v_o in self.degree.items():
                if v_o > v:
                    curr[k_o] = 1
            hyp_deg[k] = curr
        hyp_deg = adjacency_mask.multiply(hyp_deg)
            
        single_hypotheses['degree'] = csr_matrix(normalize(hyp_deg, norm='l1', axis=1))
        return [single_hypotheses, single_hypotheses, single_hypotheses, single_hypotheses]

    def create_transitions(self) -> Tuple[csr_matrix, csr_matrix]:
        print("Deriving transitions ...")
        n_transitions = sum([len(s) - 1 for s in self.filtered_index_sequences])
        transitions = np.zeros((n_transitions, 2), dtype="int16")
        index = 0
        for seq in self.filtered_index_sequences:
            transitions_in_sequence = list(zip(seq, seq[1:]))
            for t in transitions_in_sequence:
                transitions[index, ] = t
                index += 1
        print("Number of transitions:", n_transitions)

        p_gt = np.empty((n_transitions, 2))
        index = 0
        for seq in self.filtered_index_sequences:
            transitions_in_sequence = list(zip(seq, seq[1:]))
            if self.steps is None:
                curr_cut_off = int(len(seq) / 2)
            else:
                curr_cut_off = self.steps
            for _ in range(curr_cut_off):
                p_gt[index, ] = [1, 0]
                index += 1
            for _ in transitions_in_sequence[curr_cut_off:]:
                p_gt[index, ] = [0, 1]
                index += 1
        curr_transitions = [csr_matrix((self.states, self.states)), csr_matrix((self.states, self.states))]
        degree_buckets_transitions = [csr_matrix((self.bins,self.bins)), csr_matrix((self.bins,self.bins))]
        bins = self.create_bins(self.degree, self.bins)
        for i in range(len(transitions)):
            curr_transitions[0 if p_gt[i][0] == 1 else 1][transitions[i][0], transitions[i][1]] += 1
            if transitions[i][0] in bins and transitions[i][1] in bins:
                degree_buckets_transitions[0 if p_gt[i][0] == 1 else 1][bins[transitions[i][0]], bins[transitions[i][1]]] += 1
        return curr_transitions, degree_buckets_transitions

    @staticmethod
    def create_bins(data: dict, bins: int) -> dict:
        processed_data = np.array(sorted(list(data.items()), key=lambda x: x[1]))
        bins = [processed_data[:,0][int(len(processed_data)/bins*bin):int(len(processed_data)/bins*(bin+1))] for bin in range(bins)]
        return {a: i for i, b in enumerate(bins) for a in b}


    def get_transitions(self) -> list:
        if self.type == "degree":
            return self.degree_transitions
        return self.transitions

    def get_hypotheses(self) -> list:
        return self.hypotheses


if __name__ == '__main__':
    basedir = Path("data", "wikispeedia")
    for option in ["wiki_matrices"]:
        dataset = WikipediaDataset(args={
            'basedir': basedir,
            'test': False, 
            'type': 'degree',
            'bins': 10, 
            'steps': None,
            'sequence_path': Path("wikispeedia_paths-and-graph", "paths_finished.tsv"),
            'article_path': Path("wikispeedia_paths-and-graph", "articles.tsv"),
            'matrix_path': Path("wikispeedia_paths-and-graph", "shortest-path-distance-matrix.txt"),
            'link_path': Path("wikispeedia_paths-and-graph", "links.tsv"),
            'verbose': 1,
        })
        if not Path.exists(Path(basedir, option)):
            Path.mkdir(Path(basedir, option))
        save_transitions_and_hypothesis(dataset=dataset, path=Path(basedir, option))
