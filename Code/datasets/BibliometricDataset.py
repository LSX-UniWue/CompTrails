import json
import multiprocessing
import networkx as nx
import torch
import umap
from tqdm import tqdm
from transformers import *
from scipy.sparse import csr_matrix, vstack
from pathlib import Path
from functools import partial
from typing import Tuple
from collections import defaultdict
from geopy.distance import geodesic as GD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Code.datasets.AbstractDatasets import *
# from AbstractDatasets import *

# Wikidata query for universities: 
# SELECT ?universityLabel ?universityDescription ?coord WHERE {
#   ?university (wdt:P31/wdt:P279*) wd:Q3918.
#   ?university wdt:P625 ?coord.
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "en,de". }
# }

WORKER = 30

def get_geo_closeness(input_tuple: tuple) -> list:
    n, author_city, distances, city_id = input_tuple
    curr_distances = []
    n_cities = [city_id[x] for x in author_city[n].keys()]  # list of city ids of author n
    for m in author_city.keys():
        m_cities = [city_id[y] for y in author_city[m].keys()]  # list of city ids of author m
        distances = [distances[a, b] for a in n_cities for b in m_cities]
        curr_distances.append([n, m, 1 - min(distances)])
    return curr_distances

def mean_vectors(author_id_dict: dict, authorid_graph_id_dict: dict, vectors: dict, dict_to_id: dict, authors: dict, author) -> csr_matrix:
    if author in author_id_dict and author_id_dict[author] in authorid_graph_id_dict:
        curr_vector = [vectors[dict_to_id[x]] for x in authors[author] if x in dict_to_id]
        if len(curr_vector) > 1:
            curr_vector = scipy.sparse.vstack(curr_vector).mean(axis=0)
        elif len(curr_vector) == 1:
            curr_vector = curr_vector[0]
        else:
            return -1, ... 
        return author, curr_vector
    return -1, ...

def calc_distance_parallel(state_dict: dict, coords_dict: dict, item) -> float:
    curr_vector = csr_matrix((1, len(state_dict)))
    name, idx = item
    if name in coords_dict:
        for o_name, idy in state_dict.items():
            if o_name in coords_dict and idx != idy:
                distance = GD(coords_dict[name], coords_dict[o_name]).km
                curr_vector[0, idy] = distance
        return idx, curr_vector
    return idx, curr_vector

def get_similarities(author_vectors: dict, size: int, k_sims: int, input: Tuple) -> Tuple[int, csr_matrix]:
    return_csr = csr_matrix((1, size))
    if input[1].sum() != 0:
        sims = cosine_similarity(input[1].reshape(1, -1), author_vectors)[0]
        inter_idx = np.argpartition(sims, k_sims)
        for other_i in inter_idx[:k_sims]:
            return_csr[0, other_i] = sims[other_i]
    return input[0], return_csr

def mean_vectors_multiprocessing(papers_per_authors: dict, vector_repr: dict, author_to_id: dict, curr_author: str) -> Tuple[int, csr_matrix]:
    papers = papers_per_authors[curr_author]
    curr_vector = [csr_matrix(vector_repr[x]) for x in papers if x in vector_repr]
    if len(curr_vector) > 1:
        curr_vector = vstack(curr_vector).mean(axis=0)
    elif len(curr_vector) == 0:
        curr_vector = csr_matrix((1, 768))
    else: 
        curr_vector = curr_vector[0]
    return author_to_id[curr_author], curr_vector

def save_dict(filepath: str, dict_to_save: dict) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        for key, value in dict_to_save.items():
            f.write(f"{key}\t{','.join([str(x) for x in value])}\n")
    print(f"Saved to {filepath}")

def load_dict(file_path) -> dict:
    ret_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = line.split("\t")
            ret_dict[tmp[0]] = np.array(tmp[1].split(","), dtype="float16")
    return ret_dict

class BibliometricDataset(ReadWorldDataset):
    def __init__(self, args: dict):
        super().__init__(args=args)
        self.datasets = args['datasets']
        self.file_paths = [f"{dataset}_dataset.json" for dataset in self.datasets]
        self.author_files = [f"{dataset}_persons.json" for dataset in self.datasets]
        self.author_countries = [f"{dataset}_author_countries.json" for dataset in self.datasets]
        self.reload_scibert_paper_embeddings = args['reload_scibert_paper_embeddings']
        self.reload_scibert_author_embeddings = args['reload_scibert_author_embeddings']
        self.verbose = args['verbose']
        self.k_sims = args['k_sims']
        self.type = args['type']
        self.debug = args['debug']
        self.authors = {}
        self.author_id_dicts = {}
        self.transitions = []
        self.hypotheses = []
        self.authorID_graphID = [{} for _ in range(len(self.file_paths))]
        for i, file_path in enumerate(self.file_paths):
            if self.verbose == 2:
                print("Creating data_generation ", file_path, " ...")
            paper_per_authors, curr_links, abstract_per_paper = self.get_authors_and_links(Path(self.basedir, file_path))
            self.authors[file_path] = paper_per_authors
            self.author_id_dicts[file_path] = {author: i for i, author in enumerate(paper_per_authors.keys())}
            curr_links = [(self.author_id_dicts[file_path][edge[0]], self.author_id_dicts[file_path][edge[1]]) for edge in curr_links]
            if self.verbose == 2:
                print("\tCreated links for " + file_path + ". Next up is to create the graph. ")
            graph = nx.MultiGraph()
            for edge in tqdm(curr_links, desc="Adding edges to the current graph. "):
                graph.add_edge(edge[0], edge[1])
            graph = graph.subgraph(max(nx.connected_components(graph), key=len))
            print(f"DEBUG: Nodes {len(graph.nodes)} - Edges {len(graph.edges)}. ")
            for graph_id, author_id in enumerate(graph.nodes):
                self.authorID_graphID[i][author_id] = graph_id
            if self.type == "affiliation":
                print("Creating affiliation reduction. ")
                reduced_matrix, state_dict = self.extract_statetype_graph(network=nx.adjacency_matrix(graph), i=i, file_path=file_path)
                self.transitions.append(reduced_matrix)
                self.hypotheses.append(self.create_affiliation_hypotheses(state_dict))
            elif self.type == "country":
                print("Creating country reduction. ")
                reduced_matrix, state_dict = self.extract_statetype_graph(network=nx.adjacency_matrix(graph), i=i, file_path=file_path, affiliation=False)
                self.transitions.append(reduced_matrix)
                self.hypotheses.append(self.create_country_hypotheses(state_dict))
            elif self.type == "cognitive":
                print("Creating cognitive reduction. ")
                self.get_semantic_similarity_clutser(abstracts_per_paper=abstract_per_paper, file_path=file_path, i=i)
                exit(0)
            else:
                self.hypotheses.append(self.create_hypothesis(graph=graph, i=i, file_path=file_path, abstracts=abstract_per_paper))
                self.transitions.append(nx.adjacency_matrix(graph))

    def extract_statetype_graph(self, network: csr_matrix, i: int, file_path: str, affiliation: bool = True) -> csr_matrix:
        if affiliation:
            author_per_criteria = self.get_affiliation_dict(i, file_path)
        else:
            author_per_criteria = self.get_country_dict(i, file_path)
        reduced_data = csr_matrix((len(author_per_criteria), len(author_per_criteria)))
        criteria2id = {v: k for k, v in enumerate(author_per_criteria.keys())}
        rev_dict = {c: criteria2id[a] for a, b in author_per_criteria.items() for c in b}
        ind, pointer, data = network.indices, network.indptr, network.data
        for row_ind, pointer_ind in tqdm(enumerate(pointer), desc="Create transitions ..."): 
            if row_ind in rev_dict:
                row_idx = ind[pointer_ind: pointer_ind+1]
                row_data = data[pointer_ind: pointer_ind+1]
                for col_ind, col_data in zip(row_idx, row_data):
                    if col_ind in rev_dict:
                        reduced_data[rev_dict[row_ind], rev_dict[col_ind]] += col_data
        return reduced_data, criteria2id

    def create_country_hypotheses(self, state_dict: dict) -> dict:
        country2latlong = {}
        with open(Path(self.basedir, "countries.csv"), 'r') as f:
            for line in f:
                line = line.strip().split(",")
                country2latlong[line[3]] = (line[1], line[2])
        distance_hyp = csr_matrix((len(state_dict), len(state_dict)))
        for name, idx in tqdm(state_dict.items(), desc="Creating distance hypothesis for country ..."):
            for o_name, idy in state_dict.items():
                if idx != idy:
                    distance = GD(country2latlong[name], country2latlong[o_name]).km
                    distance_hyp[idx, idy] = distance
                else:
                    distance_hyp[idx, idy] = 0.0
        distance_hyp /= distance_hyp.max()
        return {
            "same_country": csr_matrix(np.diag(np.ones(len(state_dict)))), 
            "diff_country": csr_matrix(np.triu(np.ones((len(state_dict), len(state_dict)))) - np.diag(np.ones(len(state_dict)))),
            "distance": distance_hyp
        }

    def create_affiliation_hypotheses(self, state_dict: dict) -> dict:
        affiliation2latlong = {}
        with open(Path(self.basedir, "affiliations.csv"), 'r') as f:
            for line in f:
                line = line.strip().split(",")
                coords = line[-1].split(" ")
                affiliation2latlong[line[0]] = (coords[1], coords[0])
        country2latlong = {}
        with open(Path(self.basedir, "countries.csv"), 'r') as f:
            for line in f:
                line = line.strip().split(",")
                country2latlong[line[3]] = (line[1], line[2])
        combined_dict = {}
        for name in state_dict.keys():
            matched = False
            for k, v in affiliation2latlong.items():
                if k in name.strip():
                    combined_dict[name] = v
                    matched = True
            if not matched:
                if name.split(",")[-1].strip() in country2latlong:
                    combined_dict[name] = country2latlong[name.split(",")[-1].strip()]
        print(f"All keys: {len(state_dict)} and found {len(combined_dict)}")
        distance_hyp = csr_matrix((len(state_dict), len(state_dict)))
        distance_parallel = partial(calc_distance_parallel, state_dict, combined_dict)
        with multiprocessing.Pool(WORKER) as pool:
            for idx, curr_vector in tqdm(pool.imap_unordered(distance_parallel, state_dict.items()), total=len(state_dict), desc="Creating distance hypothesis for affiliations ... "):
                distance_hyp[idx] = curr_vector
        distance_hyp /= distance_hyp.max()
        return {
            "same_affiliation": csr_matrix(np.diag(np.ones(len(state_dict)))), 
            "diff_affiliation": csr_matrix(np.triu(np.ones((len(state_dict), len(state_dict)))) - np.diag(np.ones(len(state_dict)))),
            "distance": distance_hyp
        }

    @staticmethod
    def generate_scibert_embeddings_per_paper(abstracts: dict) -> dict: 
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        ret_dict = {}
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
        len_dict = defaultdict(int)
        with torch.no_grad():
            for name, abstract in tqdm(abstracts.items()):
                    bert_input = torch.tensor([tokenizer.encode(abstract, add_special_tokens=True, max_length=511, truncation=True)]).to(device)
                    len_dict[len(bert_input)] += 1
                    last_hidden_states = model(bert_input)[0]  # Models outputs are now tuples
                    ret_dict[name] = last_hidden_states[-1][0].cpu().numpy()
        return ret_dict, len_dict

    @staticmethod
    def mean_vectors(paper_per_authors: dict, vector_repr: dict) -> Tuple[csr_matrix, dict]:
        author_to_id = {name: i for i, name in enumerate(paper_per_authors.keys())}
        author_vectors = csr_matrix((len(paper_per_authors), 768))
        map_func = partial(mean_vectors_multiprocessing, paper_per_authors, vector_repr, author_to_id)
        with multiprocessing.Pool(WORKER) as pool:
            for idx, curr_vector in tqdm(pool.imap_unordered(map_func, paper_per_authors.keys()), total=len(paper_per_authors), desc="Meaning vectors. "):
                author_vectors[idx] = curr_vector
        return author_vectors, author_to_id

    def get_semantic_similarity_clutser(self, abstracts_per_paper: dict, file_path: str, i: int) -> None:
        if self.reload_scibert_paper_embeddings:
            paper_embeddings = self.generate_scibert_embeddings_per_paper(abstracts=abstracts_per_paper)
            save_dict(Path(self.basedir, f"{self.datasets[i]}_scibert_paper_embeddings.csv"), paper_embeddings)
        else:
            paper_embeddings = load_dict(Path(self.basedir, f"{self.datasets[i]}_scibert_paper_embeddings.csv"))
        if self.reload_scibert_author_embeddings:
            author_embeddings = self.mean_vectors(self.authors[file_path], paper_embeddings)
            save_dict(Path(self.basedir, f"{self.datasets[i]}_scibert_author_embeddings.csv"), paper_embeddings)
        else:
            author_embeddings = load_dict(Path(self.basedir, f"{self.datasets[i]}_scibert_author_embeddings.csv"))
        reducer = umap.UMAP()
        author_embedding_matrix = np.vstack([x for x in author_embeddings.values()])
        print(f"author_embedding_matrix: {author_embedding_matrix.shape}")
        reduced_authors = reducer.fit_tramsform(author_embedding_matrix)
        print(f"reduced_authors: {reduced_authors.shape}")
        umap_authors = {x: vector for x, vector in zip(author_embeddings.keys(), reduced_authors)}
        save_dict(Path(self.basedir, f"{self.datasets[i]}_umap_author_embeddings.csv"), umap_authors)
        # then hdbscan


    def create_hypothesis(self, graph: nx.MultiGraph, i: int, file_path: str, abstracts: dict) -> dict:
        if self.verbose:
            print(f"Extracting current transitions. ")
        curr_transitions = nx.adjacency_matrix(graph)
        adjacency_mask = curr_transitions.copy()
        adjacency_mask[adjacency_mask > 0] = 1
        print(f"DEBUG: and density: {curr_transitions.count_nonzero() / (curr_transitions.shape[0] * curr_transitions.shape[1])}")
        if self.verbose:
            print(f"Extracting affiliation hypothesis. ")
        affiliation_hypothesis = self.get_affiliation_country_hypothesis(i=i, file_path=file_path, size=len(graph.nodes))
        if self.verbose:
            print(f"Extracting country hypothesis. ")
        country_hypothesis = self.get_affiliation_country_hypothesis(i=i, file_path=file_path, size=len(graph.nodes), affiliation=False)
        if self.verbose:
            print(f"Extracting Cognitive hypothesis. ")
        cognitive_hypothesis = self.get_cognitive_hypothesis(i=i, file_path=file_path, abstracts=abstracts, size=len(graph.nodes))
        if self.verbose:
            print(f"Extracting hub hypothesis. ")
        hub_hypothesis = self.get_hub_hypothesis(i=i, graph=graph)
        return {
            'hub': hub_hypothesis,
            'affi': affiliation_hypothesis, 
            'affi_graph': affiliation_hypothesis.multiply(adjacency_mask), 
            'country': country_hypothesis, 
            'country_graph': country_hypothesis.multiply(adjacency_mask),
            'cognitive': cognitive_hypothesis, 
            'cognitive_graph': cognitive_hypothesis.multiply(adjacency_mask),
            'data': row_wise_normalize(curr_transitions)}

    def get_hub_hypothesis(self, i: int, graph: nx.MultiGraph) -> csr_matrix:
        hyp = csr_matrix((len(graph.nodes), len(graph.nodes)))
        for n in tqdm(graph.nodes, total=len(graph.nodes), desc="Creating hub hypothesis for " + str(i) + " ... "):
            neighs = np.array([(x, graph.degree[x]) for x in graph.neighbors(n)])
            probas = neighs[:, 1] / neighs[:, 1].sum() 
            choice = np.random.choice(neighs[:, 0], 1, p=probas)[0]
            hyp[self.authorID_graphID[i][n], self.authorID_graphID[i][choice]] = 1
        return row_wise_normalize(hyp)

    def get_country_dict(self, i: int, file_path: str) -> dict:
        author_per_country = defaultdict(list)
        with open(Path(self.basedir, self.author_countries[i]), 'r') as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line['author'], list):
                    curr_authors = line['author']
                else:
                    curr_authors = [line['author']]
                for curr_author in curr_authors:
                    if curr_author in self.author_id_dicts[file_path] and 'countries' in line:
                        for country in line['countries']:
                            if self.author_id_dicts[file_path][curr_author] in self.authorID_graphID[i] and isinstance(country, str):
                                author_per_country[country].append(self.authorID_graphID[i][self.author_id_dicts[file_path][curr_author]])
        print(f"Having {len(author_per_country)} affiliations. ")
        return author_per_country

    def get_affiliation_dict(self, i: int, file_path: str) -> dict:
        author_per_affiliation = defaultdict(list)
        with open(Path(self.basedir, self.author_files[i]), 'r') as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line['author'], list):
                    curr_authors = line['author']
                else:
                    curr_authors = [line['author']]
                for curr_author in curr_authors:
                    if curr_author in self.author_id_dicts[file_path] and "note" in line:
                        if isinstance(line["note"], list):
                            curr_criteria = line["note"]
                        else:
                            curr_criteria = [line["note"]]
                        for crit in curr_criteria:
                            if self.author_id_dicts[file_path][curr_author] in self.authorID_graphID[i] and isinstance(crit, str):
                                author_per_affiliation[crit].append(self.authorID_graphID[i][self.author_id_dicts[file_path][curr_author]])
        print(f"Having {len(author_per_affiliation)} affiliations. ")
        return author_per_affiliation

    def get_affiliation_country_hypothesis(self, i: int, file_path: str, size: int, affiliation: bool = True) -> csr_matrix:
        if affiliation:
            author_per_criteria = self.get_affiliation_dict(i, file_path, affiliation)
        else:
            author_per_criteria = self.get_country_dict(i, file_path)
        hyp = csr_matrix((size, size))
        for vals in author_per_criteria.values():
            if len(vals) > 1:
                for m in vals:
                    for n in vals:
                        if n != m:
                            hyp[n, m] += 1
        return row_wise_normalize(hyp)

    @staticmethod
    def get_tf_idf_vectors_per_abstract(max_features: int, abstracts: dict) -> np.array:
        vect = TfidfVectorizer(max_df=0.8, sublinear_tf=True, max_features=max_features)
        texts = list(abstracts.values())
        return vect.fit_transform(texts)

    def get_author_vectors(self, size: int, i: int, max_features: int, abstracts: dict, file_path: str) -> csr_matrix:
        vectors = self.get_tf_idf_vectors_per_abstract(max_features=max_features, abstracts=abstracts)
        dict_to_id = {name: i for i, name in enumerate(abstracts.keys())}
        author_vectors = csr_matrix((size, vectors.shape[1]))
        map_func = partial(mean_vectors, self.author_id_dicts[file_path], self.authorID_graphID[i], vectors, dict_to_id, self.authors[file_path])
        with multiprocessing.Pool(WORKER) as pool:
            for author, curr_vector in tqdm(pool.imap_unordered(map_func, self.authors[file_path]), total=len(self.authors[file_path]), desc="Creating cognitive hypothesis. "):
                if author != -1:
                    author_vectors[self.authorID_graphID[i][self.author_id_dicts[file_path][author]]] = curr_vector
        return author_vectors

    def get_cognitive_hypothesis(self, i: int, file_path: str, abstracts: dict, size: int) -> csr_matrix:
        author_vectors = self.get_author_vectors(size=size, i=i, max_features=1000, abstracts=abstracts, file_path=file_path)
        hyp = csr_matrix((size, size))
        map_func = partial(get_similarities, author_vectors, size, self.k_sims)
        with multiprocessing.Pool(WORKER) as pool:
            for idx, vector in tqdm(pool.imap_unordered(map_func, [(i, x) for i, x in enumerate(author_vectors)]), total=author_vectors.shape[0], desc="Calculating pairwise sim ..."):
                hyp[idx] = vector
        return hyp, author_vectors

    @staticmethod
    def get_authors_and_links(file_path: str) -> Tuple[dict, list, dict]:
        paper_per_authors = defaultdict(list)
        abstract_per_paper = {}
        links = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                pub = json.loads(line)
                curr_authors = pub['author'] if isinstance(pub['author'], list) else [pub['author']]
                for author in curr_authors:
                    paper_per_authors[author].append(pub['key'])
                links.extend([(curr_authors[i], curr_authors[k]) for i in range(len(curr_authors)) for k in range(len(curr_authors)) if i != k])
                if 'abstract' in pub:
                    abstract_per_paper[pub['key']] = pub['abstract']
        return paper_per_authors, links, abstract_per_paper

    def get_transitions(self) -> list:
        return self.transitions

    def get_hypotheses(self) -> list:
        return self.hypotheses


if __name__ == '__main__':
    datasets = ["ss", "dm", "ai", "sp", "r", "hci"]
    all_datasets = ["ss", "dm", "ai", "sp", "r", "hci"]
    basedir = Path("data", 'bibliometric')
    dataset_type = "cognitive"
    dataset = BibliometricDataset(args={
        'basedir': basedir, 
        'datasets': datasets,
        'type': dataset_type, 
        'k_sims': 100, 
        'debug': False, 
        'reload_scibert_paper_embeddings': True,
        'reload_scibert_author_embeddings': True,
        'verbose': 1 
        })
    evaluate_dataset(dataset=dataset)
    path = str(Path(basedir, "bibliometric_matrices"))
    if dataset_type == "country":
        path = f"{path}_country"
    elif dataset_type == "affiliation":
        path = f"{path}_affiliation"
    if not os.path.exists(path):
        os.mkdir(path)
    if len(datasets) == 1:
        index = all_datasets.index(datasets[0])
    else:
        index = None
    save_transitions_and_hypothesis(dataset=dataset, path=path, start_idx=index)
