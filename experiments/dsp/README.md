# Lean Draft Sketch Prove (DSP)

based on Sean Welleck's DSP for Isabelle: https://github.com/wellecks/ntptutorial/tree/main/partII_dsp

## Execution

First of all, build the experiment repo.

``` sh
# experiments/dsp
cd lean_src_proj
lake build
```
Then run `main.py`
``` sh
python3 main.py -h
```

The main command for running DSP is `eval`. Due to the multitude of data format
out there, use the `--format` flag to specify the data format. For example,
running DSP on minif2f is:

``` sh
python3 main.py eval \
    --dataset ../minif2f/valid.jsonl \
    --format minif2f \
    --output results-minif2f-valid
```

Then, use `plot.py` to generate the plots

``` sh
python3 plot.py \
    --result results-minif2f-{valid,test} \
    --names valid test \
    --plot-output output-plot
```

## Related work

### Tony's AF
Ton'y original AF: ((Yuhuai et al.))[https://arxiv.org/abs/2205.12615]
Tony's paper improve MiniF2F from: `29.6% to 35.2%`, by `5.6%`. 

Expert Iteration:
-  AF used: "We explore if one can improve neural theorem provers by training the neural models on proofs of automatically translated theorems".
    - they only translate **problem theorems** (nl_thm := "problem + answer") then use a prover to get the formal proof.
- ExpIt algorithn:
    - `M_0 := Isabelle_Thor()`
    - `Search/Prover := Best_First_Search()`  # TODO recall best first search
    - ExpIT.fine_tune := "train model to predict next proof_step/tactic given current proof_state and previous proof_step on successful proofs.
        - i.e., `<x=(proof_state_{t}, proof_step_{t-1}), y=(proof_step_{t})>`  #TODO: I think, confirm with Albert https://twitter.com/messages/1253358235-1267913180153548800

Base Model for Neural Theorem Prover (NTP):
- Thor_GPT2 := "We use a pre-trained and fine-tuned Thor based on a GPT-2 with 700M non-embedding parameters." Note: ReProver used 299M parameters enc-dec. 
- fine-tuned on the PILE arxiv + github

Neural Theorem Prover (NTP) for `M_0`:
- Thor := 
    - The Thor agent is fine-tuned on the PISA dataset which consists of 2.49 million proof steps from the Isabelle/HOL library.
    - The model is trained with the objective to predict the next token in va proof step, given the proof state and the last proof step.
    - proof step := "tactic in Isabelle"  #TODO confirm with Albert https://twitter.com/messages/1253358235-1267913180153548800

Questions: 
- Q1: what is this: "we perform deduplication by problem statements" when does it matter? All MATH train are unique, so why would I care about this?

Idea:
- Idea1: use the formal ground truth solution string in MATH, implement Draft Sketch Proof (DSP) for Lean4 + use some symbolic/ntp solver (hammer/tidy/ReProver)
