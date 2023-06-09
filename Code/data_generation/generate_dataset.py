import urllib.request
import shutil
import gzip
import json
import re
import sys

from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from dblp_parser import parse_dblp, parse_dblp_person, get_dblp_country


DATA_PATH = Path('data', 'bibliometric')  # local "../data/"
DBLP_URL = 'https://dblp.org/xml/'
SEMANTIC_SCHOLAR_URL = 'htthttps://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2022-05-01/'


def download_dblp() -> None:
    """
    This functions downloads the DBLP XML and DTD.
    """
    source_gz = DBLP_URL + 'release/dblp-2020-01-01.xml.gz'
    source_dtd = DBLP_URL + 'release/dblp-2019-11-22.dtd'
    target_gz = DATA_PATH + 'dblp.xml.gz'
    target_dtd = DATA_PATH + 'dblp-2019-11-22.dtd'

    print('Downloading file ' + source_gz)
    with urllib.request.urlopen(source_gz) as response, open(target_gz, 'wb') as fh:
        shutil.copyfileobj(response, fh)
    print('Downloading file ' + source_dtd)
    with urllib.request.urlopen(source_dtd) as response, open(target_dtd, 'wb') as fh:
        shutil.copyfileobj(response, fh)
    print('Download finish!')
    print()


def unzip_dblp() -> None:
    """
    This functions unzip the DBLP dataset.
    """
    source = DATA_PATH + 'dblp.xml.gz'
    target = DATA_PATH + 'dblp.xml'

    with gzip.open(source, 'rb') as f_in:
        with open(target, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print()


def extract_publications() -> None:
    """
    Reading the DBLP XML and parse it into a json file for further processing.
    """
    source = DATA_PATH + 'dblp.xml'
    target = DATA_PATH + 'dblp.json'

    print('Parsing ' + source)
    parse_dblp(source, target)
    print('Parse finish! File dblp.json created!')
    print()


def extract_ai_publications(source_venues: Path = Path(DATA_PATH, 'ai_venues.json'),
                            source: Path = Path(DATA_PATH, 'dblp.json'),
                            target: Path = Path(DATA_PATH, 'ai_dblp.json')) -> set:
    """
    Extracting the AI publications from the DBLP dataset.
    :param source_venues:
    :param source: From where to read the pubs.
    :param target: Where to write the pubs to.
    :return: List of authors, which have published an AI publication
    """

    with open(source_venues, "r", encoding="utf-8") as f:
        venues = json.load(f)

    authors = set()
    with open(target, "w", encoding="utf-8") as out_f:
        with open(source, "r", encoding="utf-8") as in_f:
            for line in tqdm(in_f):
                line = json.loads(line)
                if line['booktitle']:
                    curr_venue = line['booktitle'][0]
                elif line['journal']:
                    curr_venue = line['journal'][0]
                else:
                    continue
                curr_venue = re.sub(" \\([0-9]+\\)$", "", curr_venue)
                if curr_venue in venues:
                    line['venue'] = curr_venue
                    json.dump(line, out_f)
                    out_f.write("\n")
                    authors.update(line['author'])
    print('Parse finish! File created!')
    print()
    return authors


def extract_persons(author_list: list,
                    source: Path = Path(DATA_PATH, 'dblp.xml'),
                    target: Path = Path(DATA_PATH, 'persons.json')) -> None:
    """
    Extracting person information from DBLP and write it down.
    Note that we exclude person records that have the
    publtype disambiguation.
    :param author_list: A list of authors,
    :param source:
    :param target:
    which have published an AI publication
    """
    print('Parsing ' + source.name)
    parse_dblp_person(source, target, author_list)
    print('Parse finish! File persons.json created!')
    print()


def parse_countries(source_person: Path = Path(DATA_PATH, 'persons.json'),
                    source_country: Path = Path(DATA_PATH, 'poss_countries.txt'),
                    target: Path = Path(DATA_PATH, 'author_countries.json')):
    """
    Extracting countries for all authors in the AI DBLP.
    Countries are found using a possible country file.
    :param source_person:
    :param source_country:
    :param target:
    """
    get_dblp_country(source_person, source_country, target)
    print('Parse finish! File author_countries.json created!')
    print()


# --- Helper for Extract Semantic scholar ---
def download_semantic_scholar_if_needed(semantic_scholar_path: Path,
                                        default_count: int = 181) -> None:
    """
    Helper file for match semantic scholar. Downloads the whole corpus.
    """
    if not Path.exists(semantic_scholar_path):
        Path.mkdir(semantic_scholar_path)
        with urllib.request.urlopen(SEMANTIC_SCHOLAR_URL + "manifest.txt") as response:
            with open(Path(semantic_scholar_path, "manifest.txt"), 'wb') as fh:
                shutil.copyfileobj(response, fh)
        with open(Path(semantic_scholar_path, "/manifest.txt"), "r") as f:
            for line in tqdm(f, total=default_count):
                line = line.strip()
                with urllib.request.urlopen(SEMANTIC_SCHOLAR_URL + line) as response:
                    with open(Path(semantic_scholar_path, line), 'wb') as fh:
                        shutil.copyfileobj(response, fh)


def get_doi(line) -> str:
    """
    Get doi for a given line of the data, useful for semantic_scholar matching"
    """
    if "ee" in line:
        for x in line["ee"]:
            if "doi" in x:
                return x.replace("https://doi.org/", "")


def match_semantic_scholar(source: Path = Path(DATA_PATH, 'ai_dblp.json'),
                           target: Path = Path(DATA_PATH, 'ai_dataset.json'),
                           semantic_scholar_path: Path = Path(DATA_PATH, "semantic_scholar/"),
                           download_sem_schol: bool = False) -> None:
    """
    Match all the publications to Semantic Scholar. Also downloads Semantic Scholar
    if needed. Writes the matched data to ai_community_dataset.json
    :param source:
    :param target:
    :param semantic_scholar_path:
    :param download_sem_schol:
    """
    if download_sem_schol:
        download_semantic_scholar_if_needed(semantic_scholar_path)

    with open(source, "r", encoding="utf-8") as f:
        pubs = f.readlines()
    pubs = [json.loads(x) for x in pubs]
    removed_indices = set()
    titles = defaultdict(list)
    [titles[x['title'].strip(".").lower()].append(i)
     for i, x in enumerate(pubs)]
    files = [str(file_path) for file_path in semantic_scholar_path.iterdir() if "s2-corpus-" in str(file_path)]
    counter = 1
    with open(target, 'w', encoding="utf-8") as out_f:
        for file_path in files:
            print(f"Reading file {file_path} ... ({counter}/{len(files)})")
            with gzip.open(file_path, 'rt', encoding="utf-8") as in_f:
                for line in in_f:
                    line = json.loads(line)
                    curr_title = line['title'].strip().lower()
                    if curr_title in titles:
                        index = None
                        for i in titles[curr_title]:
                            pub = pubs[i]
                            doi = get_doi(pub)
                            if doi and "doi" in line and line["doi"]:
                                if doi == line["doi"]:
                                    index = i
                                    break
                            elif "year" in line and line['year'] is not None and int(pub["year"]) == int(line["year"]):
                                if line["venue"] == "ArXiv":
                                    if pub["journal"] and pub["journal"][0] == "CoRR":
                                        index = i
                                        break
                                elif pub["journal"] and pub["journal"][0] == "CoRR":
                                    continue
                                else:
                                    index = i
                                    break
                        if index and index not in removed_indices:
                            if 'abstract' not in pub:
                                pub['abstract'] = line['paperAbstract']
                            if 'in_citations' not in pub:
                                pub['in_citations'] = line['inCitations']
                            if 'out_citations' not in pub:
                                pub['out_citations'] = line['outCitations']
                            if 'ss_id' not in pub:
                                pub['ss_id'] = line['id']
                            if 'doi' not in pub and 'doi' in line:
                                pub['doi'] = [line['doi']]
                            json.dump(pub, out_f)
                            out_f.write("\n")
                            removed_indices.add(index)
            counter += 1
        for i, pub in enumerate(pubs):
            if i not in removed_indices:
                json.dump(pub, out_f)
                out_f.write("\n")
    print("Finished. ")


def extract_german_ai(source: Path = Path(DATA_PATH, 'ai_community_dataset.json'),
                      target: Path = Path(DATA_PATH, 'german_ai_community_dataset.json')) -> None:
    """
    Extracts all publications in which at least one author is flagged as
    german.
    """
    countries = DATA_PATH + 'author_countries.json'
    with open(countries, "r", encoding="utf-8") as f:
        countries = f.readlines()
    countries = [json.loads(x) for x in countries]
    german_authors = [(x['author'], x['key']) for x in countries
                      if "Germany" in x["countries"]]
    german_names = {}
    for author, dblp_id in german_authors:
        if isinstance(author, list):
            for aut in author:
                german_names[aut] = dblp_id
        elif isinstance(author, str):
            german_names[author] = dblp_id
    with open(source, "r", encoding="utf-8") as in_f:
        with open(target, "w", encoding="utf-8") as out_f:
            for line in tqdm(in_f):
                line = json.loads(line)
                german_as = [auth for auth in line["ai_authors"]
                             if auth in german_names]
                if german_as:
                    line["german_ai_authors"] = german_as
                    line["german_ai_authors_keys"] = [german_names[name]
                                                      for name in german_as]
                    json.dump(line, out_f)
                    out_f.write("\n")
    print("Finished extracting german AI publications. ")


if __name__ == '__main__':
    input_args = sys.argv[1]
    datasets = []
    print(f"Starting pipeline for {input_args}")
    for abbrev, long_name in [
        ("ai", "artificial_intelligence"),
        ("dm", "data_mining"), 
        ("hci", "human_computer_interaction"), 
        ("r", "robotics"), 
        ("sp", "signal_processing"), 
        ("ss", "software_systems"),
        ("ais", "artificial_intelligence_small"),
        ("sps", "signal_processing_small")
        ]:
        if abbrev == input_args:
            datasets.append((abbrev, long_name))

    print('**** Starting pipeline process to create AI Datasets****')
    print()
    if not Path.is_dir(Path(DATA_PATH)):
        Path.mkdir(Path(DATA_PATH))

    print('Process 01 - Download dblp data')
    # download_dblp()

    print('Process 02 - Unzipping dblp data')
    # unzip_dblp()

    print('Process 03 - Create dblp.json')
    # extract_publications()

    for dataset in datasets:
        print(f'Process 04 - Extract {dataset[1].replace("_", " ").title()} ({dataset[0]}) publications ')
        author_set = extract_ai_publications(source_venues=Path(DATA_PATH, f"{dataset[1]}_venues.json"), target=Path(DATA_PATH, f"{dataset[0]}_dblp.json"))

        print('Process 05 - Create persons.json')
        extract_persons(author_list=list(author_set), target=Path(DATA_PATH, f"{dataset[0]}_persons.json"))

        print('Process 06 - Create author_countries.json')
        parse_countries(source_person=Path(DATA_PATH, f"{dataset[0]}_persons.json"), target=Path(DATA_PATH, f"{dataset[0]}_author_countries.json"))

        print('Process 07 - Extract Semantic scholar information for the AI community.')
        match_semantic_scholar(source=Path(DATA_PATH, f'{dataset[0]}_dblp.json'), target=Path(DATA_PATH, f'{dataset[0]}_dataset.json'),)

    print('*** Pipeline process to create the data sets finished! ***')
