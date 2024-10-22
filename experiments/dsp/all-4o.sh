#!/bin/bash

IMAGE=chrysoberyl/pantograph
BASE_DIR=/data/result/4o
main() {
	docker run --rm -it \
		--volume $PWD:/data \
		--entrypoint poetry \
		--env OPENAI_API_KEY=$OPENAI_API_KEY \
		$IMAGE run \
		python /data/experiments/dsp/main.py eval $@
}
main --dataset /data/experiments/dsp/ --dataset /data/experiments/minif2f/valid.jsonl --format minif2f --output $BASE_DIR/valid1
main --dataset /data/experiments/dsp/ --dataset /data/experiments/minif2f/valid.jsonl --n-samples 3 --format minif2f --output $BASE_DIR/valid3
main --dataset /data/experiments/dsp/ --dataset /data/experiments/minif2f/test.jsonl --format minif2f --output $BASE_DIR/test1
main --dataset /data/experiments/dsp/ --dataset /data/experiments/minif2f/test.jsonl --n-samples 3 --format minif2f --output $BASE_DIR/test3
