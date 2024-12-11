# Artefact Packing

This is about how to pack the artefacts described in `README.md`.

## Execute the DSP experiments

Run the DSP experiments from the project root,
```sh
python experiments/dsp/main.py eval \
	--dataset experiments/minif2f/valid.jsonl --format minif2f --output result/ref-4o/valid1
python experiments/dsp/main.py eval \
	--dataset experiments/minif2f/valid.jsonl --format minif2f --n-samples 3 --output result/ref-4o/valid3
python experiments/dsp/main.py eval \
	--dataset experiments/minif2f/test.jsonl --format minif2f --output result/ref-4o/test1
python experiments/dsp/main.py eval \
	--dataset experiments/minif2f/test.jsonl --format minif2f --n-samples 3 --output result/ref-4o/test3

python experiments/dsp/main.py eval \
	--dataset experiments/minif2f/valid.jsonl --format minif2f --model o1-preview --output result/ref-o1/valid1
python experiments/dsp/main.py eval \
	--dataset experiments/minif2f/test.jsonl --format minif2f --model o1-preview --output result/ref-o1/test1
```
Then, pack it with the script

```sh
bash experiments/dsp/pack-artefact.sh
```

