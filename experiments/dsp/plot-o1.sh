#!/usr/bin/env sh

IMAGE=chrysoberyl/pantograph
BASE_DIR=/data/result/o1

plot() {
	docker run --rm -it \
		--volume $PWD:/data \
		--entrypoint poetry \
		--env OPENAI_API_KEY=$OPENAI_API_KEY \
		$IMAGE run \
		python /data/experiments/dsp/plot.py $@
}
plot --result $BASE_DIR/{valid1,test1} --names valid1 test1 --plot-output $BASE_DIR/plot
