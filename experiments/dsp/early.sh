#!/bin/bash

IMAGE=chrysoberyl/pantograph

docker run --rm -it \
	--volume $PWD:/data \
	--entrypoint poetry \
	--env OPENAI_API_KEY=$OPENAI_API_KEY \
	$IMAGE run \
	python /data/experiments/dsp/main.py eval --output /data/result/debug
