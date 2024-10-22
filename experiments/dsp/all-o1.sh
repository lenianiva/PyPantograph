#!/bin/bash

IMAGE=chrysoberyl/pantograph

main() {
	docker run --rm -it \
		--volume $PWD:/data \
		--entrypoint poetry \
		--env OPENAI_API_KEY=$OPENAI_API_KEY \
		$IMAGE run \
		python /data/experiments/dsp/main.py eval $@
}
plot() {
	docker run --rm -it \
		--volume $PWD:/data \
		--entrypoint poetry \
		--env OPENAI_API_KEY=$OPENAI_API_KEY \
		$IMAGE run \
		python /data/experiments/dsp/plot.py $@
}
BASE_DIR=/data/result/o1
main --dataset /data/experiments/dsp/ --dataset /data/experiments/minif2f/valid.jsonl --format minif2f --model o1-preview --output $BASE_DIR/valid1
main --dataset /data/experiments/dsp/ --dataset /data/experiments/minif2f/test.jsonl --format minif2f --model o1-preview --output $BASE_DIR/test1
plot --result $BASE_DIR/{valid1,test1} --names valid1 test1 --plot-output $BASE_DIR/plot
