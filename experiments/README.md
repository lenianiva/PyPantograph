# Experiments

## Environment

The experiments require
- Internet access
- At least 15G of disk space on the drive hosting this repository
- An [OpenAI API](https://openai.com/index/openai-api/) key

## Building the container

If you already have the docker image, you can skip this step.

Execute, in the root directory of this repository,
``` sh
docker build . --tag pantograph
```

## Experiments

### Early Evaluation

Due to the nature of this project and how Lean's build system works, the first
pass needs to build the Mathlib library. This building of the library will take
about 30 minutes to run.

Execute in the project root directory
``` sh
experiments/dsp/early.sh
```
the results will be written to `result/`

