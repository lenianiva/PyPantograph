#!/bin/bash

# Packs the artefact
# Execute this at the project root directory to produce `artefact.zip`

echo "Packing Git archive"
git archive --format zip --output artefact.zip HEAD
echo "Packing docker image"
docker image save pantograph > pantograph.tar
echo "Adding experimental results"
zip -ur artefact.zip result/ref-4o result/ref-o1 pantograph.tar
