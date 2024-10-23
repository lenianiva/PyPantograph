# Experiments

Unpack the `artefact.zip` file. Then cd into the unpacked directory and execute
the commands below.

## Environment

The experiments require
- Internet access
- At least 15G of disk space on the drive hosting this repository
- An [OpenAI API](https://openai.com/index/openai-api/) key, for invoking the models.

Set the `OPENAI_API_KEY` environment variable to the API key.

### Building/Loading the container

There are 3 ways to load the container. Choose one.
- (Artefact reviewer) If you **have** the docker image:
``` sh
docker load --input pantograph.tar
```
- Use the docker repository
``` sh
docker pull chrysoberyl/pantograph
```
- Build the docker image from scratch
``` sh
docker build . --tag chrysoberyl/pantograph
```

## Experiments

The experiments are bound by limitations of OpenAI. Since OpenAI as a commercial
company cannot indefinitely store all snapshots of their models, the experiments
rely on OpenAI's provided version of `o1-preview` and `gpt-4o` models. This may
impact the reproducibility of the experiments in the future.

### Early Evaluation

Due to the nature of this project and how Lean's build system works, the first
pass needs to build the Mathlib library. This building of the library will take
about 30 minutes to run.

Execute in the project root directory
``` sh
experiments/dsp/early.sh
```
the results will be written to `result/`

### Plots for the paper

To generate the plots for the paper, execute

``` sh
bash experiments/dsp/ref-plots.sh
```

which will output the plots in `result/ref-{o1,4o}/plot` based on the provided
experiment result data in `result/ref-*`

### GPT-4o experiment

``` sh
bash experiments/dsp/all-4o.sh
bash experiments/dsp/plot-4o.sh
```

which will output the plots in `result/4o/plot`

### o1-preview experiment

``` sh
bash experiments/dsp/all-o1.sh
bash experiments/dsp/plot-o1.sh
```

which will output the plots in `result/o1/plot`
