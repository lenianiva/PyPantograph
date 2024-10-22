#!/usr/bin/env sh

IMAGE=chrysoberyl/pantograph
plot() {
	docker run --rm -it \
		--volume $PWD:/data \
		--entrypoint poetry \
		--env OPENAI_API_KEY=$OPENAI_API_KEY \
		$IMAGE run \
		python /data/experiments/dsp/plot.py $@
}

BASE_DIR=/data/result/ref-4o
plot --result $BASE_DIR/{valid1,test1,valid3,test3} --names valid1 test1 valid3 test3 --plot-output $BASE_DIR/plot
BASE_DIR=/data/result/ref-o1
plot --result $BASE_DIR/{valid1,test1} --names valid1 test1 --plot-output $BASE_DIR/plot
