#!/bin/bash

# Packs the artefact
# Execute this at the project root directory to produce `artefact.zip`

echo "Packing Git archive"
git-archive-all artefact.zip
echo "Packing docker image"
docker image save chrysoberyl/pantograph > pantograph.tar
echo "Adding experimental results"
zip -ur artefact.zip result/ref-4o/{valid1,valid3,test1,test3} result/ref-o1/{valid1,valid3} pantograph.tar
