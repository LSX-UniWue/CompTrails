#!/usr/bin/env bash
# docker build -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/comptrails-dataset:latest -f Code/data_generation/Dockerfile .
# docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/comptrails-dataset:latest
# buildah bud -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/comptrails-dataset:latest -f Code/data_generation/Dockerfile .
# buildah push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/comptrails-dataset:latest
for dataset in "sps" "ais"; do
    sed -e 's/{{}}/'"$dataset"'/' kubernetes/extract_dataset/dataset.yml >> tmp.yml
    kubectl -n koopmann delete job comptrails-$dataset-dataset
    kubectl -n koopmann create -f tmp.yml
    rm tmp.yml
done