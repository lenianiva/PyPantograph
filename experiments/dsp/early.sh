#!/bin/bash

docker run --rm -it \
	--volume $PWD:/data \
	--entrypoint poetry \
	--env OPENAI_API_KEY=$OPENAI_API_KEY \
	pantograph run \
	python experiments/dsp/main.py eval --output /data/result/debug
