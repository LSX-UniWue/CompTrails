# CompTrails

This is the repository for the publication "CompTrails: Comparing Hypotheses across Behavioral Networks". 

Behavioral Networks is a collective term for networks that contain relational information on human behavior. This ranges from social networks that contain friendships or cooperations between individuals, to navigational networks that contain geographical or web navigation, and many more. Understanding the forces driving behavior within these networkscan be beneficial in improving the underlying network, for example, by generating new hyperlinks on websites or proposing new connections and friends in social networks. Previous approaches considered different hypotheses on these networks and evaluated which hypothesis best fits the network. These hypotheses can represent human intuition, expert opinions, or be based on previous insights. In this work, we extend these approaches to enable the comparison of a single hypothesis across different networks. By understanding the differences in how humans navigate within different networks, it is possible to adjust the networks’ structures and create more intuitive and user-friendly experiences. Furthermore, such comparisons can reveal best practices and identify areas for improvement between different networks. We show that naive comparisons do not work and unveil several issues that potentially impact comparisons and lead to undesired results. Based on these findings, we propose a framework with five optional components. Each is applicable in different settings and enables addressing specific analysis goals. We show the benefits of our approach by applying it to synthetic data and several real-world datasets, including web navigation, bibliographic navigation, and geographic navigation.

## Synthetic Example

Simple run the script `SyntheticExample.py` in the folder `Code/`. The script will generate a synthetic dataset and run the algorithm on it. The results will be saved in the folder `data/synthetic/` and the final images under `data/images`.

## Synthetic data
Synthetic data will be generated using the respective dataset under `Code/datasets/BarabasiDatasets.py`. 
Here, different settings can be implemented to generate the different data from the paper. 
Otherwise, the respective setting can be applied in the main file `CompTrails.py`, where the data is generated and the algorithm is run in one sweep.

## Real World Data
Here a small description of the data used in this project and how to access them. 

### Bibliographic

This dataset is created very similar to the dataset at [Zenodo](https://zenodo.org/records/3930390). 
It can be constructed using the script `generate_dataset.py` in the folder `Code/data_generation`. The script will download the data from the DBLP website and save it in the folder `data/bibliographic/`. 
Since we match the DBLP dataset with semantic scholar, and it is very likely that the data is not the same as in the original dataset, we provide the final data in this repository. 

### Wikispeedia
You can download them here: https://snap.stanford.edu/data/wikispeedia.html 

Unpack them in 'data/wikispedia/'. The script `WikispeediaDataset.py` in the folder `Code/datasets` will then generate the respective transitions and matrices. The results will be saved in the folder `data/wikispeedia/` and the final images under `data/images`. 

### Flickr

These data can originally be requested from the authors of the paper "Photowalking the city:
Comparing hypotheses about urban photo trails on Flickr" by Becker et al. (2015).
Otherwise, we uploaded the 6 used cities in this repository. 
The script `FlickrDataset.py` in the folder `Code/datasets` will then generate the respective transitions and matrices. The results will be saved in the folder `data/flickr/` and the final images under `data/images`. 

