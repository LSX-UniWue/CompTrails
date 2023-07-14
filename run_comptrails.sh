#!/usr/bin/env bash

# docker buildx build --platform linux/amd64 --load -t  ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/comptrails:latest -f kubernetes/comptrails/Dockerfile .
# docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann/comptrails:latest

for dataset in "loaded-bib"; do
  sed -e 's/{dataset}/'"$dataset"'/' kubernetes/comptrails/comptrails.yml >> tmp.yml
  kubectl -n koopmann delete job comptrails-$dataset
  kubectl -n koopmann create -f tmp.yml
  rm tmp.yml
done
