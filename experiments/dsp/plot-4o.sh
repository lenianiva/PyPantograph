#!/usr/bin/env sh

IMAGE=chrysoberyl/pantograph
BASE_DIR=/data/result/4o

plot() {
	docker run --rm -it \
		--volume $PWD:/data \
		--entrypoint poetry \
		--env OPENAI_API_KEY=$OPENAI_API_KEY \
		$IMAGE run \
		python /data/experiments/dsp/plot.py $@
}
plot --result $BASE_DIR/{valid1,valid3,test1,test3} --names valid1 valid3 test1 test3 --plot-output $BASE_DIR/plot
