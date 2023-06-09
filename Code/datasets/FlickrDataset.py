import json
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from collections import defaultdict
from Code.datasets.AbstractDatasets import *


# To remember: http://geodb-cities-api.wirefreethought.com/docs/tutorials/wikidata/city-tourist-attractions

class FlickrDataset(ReadWorldDataset):
    def __init__(self, args: dict):
        super(FlickrDataset, self).__init__(args=args)
        self.files = args['files']
        self.data = []
        self.user = {}
        self.trails = {}
        self.attractions = {}
        self.attraction_closeness = {}
        self.geo_coords = {}
        self.transitions = []
        self.hypotheses = []
        for file in self.files:
            data, user, trails, geo_coords = self.load_file(file=file)
            self.data.append(data)
            self.user[file] = user
            self.trails[file] = trails
            self.attractions[file] = self.extract_attractions(file_path=file)
            self.attraction_closeness[file] = self.get_nearest_attraction(geo_coords, self.attractions[file], city=file)
            self.geo_coords[file] = {'coordinates': geo_coords[:, [1, 2]],
                                     'mean': geo_coords[:, [1, 2]].mean(axis=0),
                                     'max': np.max(geo_coords[:, [1, 2]], axis=0),
                                     'min': np.min(geo_coords[:, [1, 2]], axis=0)
                                     }
            self.transitions.append(self.map_trails_to_matrix(trails, attractions=self.attraction_closeness[file]))
            self.hypotheses.append(self.create_hypothesis(self.geo_coords[file], self.attractions[file]))
        print("Finished. ")

    def load_file(self, file: str) -> Tuple[list, set, dict, list]:
        geo_coords = []
        curr_data = []
        user = set()
        trails = defaultdict(list)
        with open(f"{Path(str(self.basedir), file)}.csv", 'r', encoding='ISO-8859-1') as f:
            for i, line in tqdm(enumerate(f)):
                line = line.strip().split(",")
                curr_data.append(line)
                user.add(line[1])
                diff = len(line) - 13
                geo_coords.append([i, float(line[6 + diff]), float(line[7 + diff])])
                trails[line[1]].append([i, line[3], float(line[6 + diff]), float(line[7 + diff]), line[5]])
        return curr_data, user, trails, np.array(geo_coords)

    @staticmethod
    def simple_distance(coord1: tuple, coord2: tuple) -> float:
        return pow(pow(coord1[0] - coord2[0], 2) + pow(coord1[1] - coord2[1], 2), 0.5)

    def get_nearest_attraction(self, coordinates: np.array, attractions: dict, city: str) -> list:
        res_list = []
        for coord in tqdm(coordinates, desc=f"Calculating geo closeness for {city}"):
            distances = [(att_id, self.simple_distance(coord[1:], att_coord[1])) for att_id, att_coord in attractions.items()]
            distances = sorted(distances, key=lambda x: x[1])
            res_list.append(distances[0][0])  # possible save geo_coord id with it
        return res_list

    def extract_attractions(self, file_path) -> dict:
        def get_lat_long_from_point(point_str: str) -> Tuple[float, float]:
            tmp = point_str.split("(")[1][:-1].strip().split(" ")
            return float(tmp[1]), float(tmp[0])
        with open(f'{Path(self.basedir, "attractions", file_path)}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = {i: (x['attractionLabel'], get_lat_long_from_point(x['gps'])) for i, x in enumerate(data)}
        return data

    def map_trails_to_matrix(self, trails: dict, attractions: list) -> csr_matrix:
        if self.verbose:
            print(f"{max(attractions) + 1} attractions in the city. ")
        transitions = csr_matrix((max(attractions) + 1, max(attractions) + 1))
        for trail in tqdm(trails.values(), desc="Creating transitions... "):
            edges = [(trail[x][0], trail[x + 1][0]) for x in range(len(trail) - 1)]
            for edge in edges:
                transitions[attractions[edge[0]], attractions[edge[1]]] += 1
        return transitions

    def map_trails_to_binary_center(self, trails: dict, geo_coords: dict, attractions: dict, attraction_closeness: list) -> csr_matrix:
        if self.verbose:
            print("Creating binary hypotheses. ")
        distance_probas = {k: v for k, v in self.calc_one_element(geo_coords['mean'], attractions)}
        median_distance = np.median([x for x in distance_probas.values()])
        transitions = csr_matrix((2, 2))
        for trail in tqdm(trails.values(), desc="Creating transitions... "):
            edges = [(trail[x][0], trail[x + 1][0]) for x in range(len(trail) - 1)]
            for edge in edges:
                transitions[0 if distance_probas[attraction_closeness[edge[0]]] < median_distance else 1, 0 if distance_probas[attraction_closeness[edge[1]]] < median_distance else 1] += 1
        return transitions

    def calc_one_element(self, coords: tuple, attractions):
        distances_to_element = np.array([(i, self.simple_distance(coords, x[1])) for i, x in attractions.items()])
        max_center_element = max(distances_to_element[:, 1])
        return [(x[0], (max_center_element - x[1]) / max_center_element) for x in distances_to_element]

    def create_hypothesis(self, geo_coords: dict, attractions: dict) -> dict:
        if self.verbose:
            print("Creating hypotheses. ")
        center_hypothesis = csr_matrix((len(attractions), len(attractions)))
        distance_hypothesis = csr_matrix((len(attractions), len(attractions)))
        distance_probas = self.calc_one_element(geo_coords['mean'], attractions)
        for i in range(center_hypothesis.shape[0]):
            for distance in distance_probas:
                center_hypothesis[i, int(distance[0])] = distance[1]
        for attraction_id, attraction in attractions.items():
            distance_probas = self.calc_one_element(attraction[1], attractions)
            for distance in distance_probas:
                distance_hypothesis[attraction_id, int(distance[0])] = distance[1]
        return {"center_distance": center_hypothesis,
                "distance": distance_hypothesis,
                "center_prob": row_wise_normalize(np.multiply(distance_hypothesis, center_hypothesis))}

    def get_attractions(self) -> dict:
        return self.attractions

    def get_coords(self) -> dict:
        return self.geo_coords

    def get_transitions(self) -> list:
        return self.transitions

    def get_hypotheses(self) -> list:
        return self.hypotheses


if __name__ == '__main__':
    basedir = Path("data", "flickr-data")
    print(f"Start creating twitter dataset for all. ")
    dataset = FlickrDataset(args={
        'basedir': basedir,
        'files': ["vancouver", "washington", "la", "london", "nyc"],
        'verbose': 1
    })
    print("Evaluate this. ")
    # evaluate_dataset(dataset=dataset)
    if not Path.exists(Path(basedir, "matrices")):
        os.mkdir(Path(basedir, "matrices"))
    save_transitions_and_hypothesis(dataset=dataset, path=Path(basedir, "matrices"))
