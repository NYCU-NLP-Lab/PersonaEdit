# PersonaEdit

PersonaEdit pipeline:

1. Select edit examples.
2. Run AlphaEdit editing.
3. Evaluate edited weights.
4. Compute metrics.

Set up an environment and install dependencies:

```bash
conda create -n personaedit python=3.10
conda activate personaedit
pip install -r requirements.txt
```

## Available Splits

### Clustered Split

Input:

```text
data/splits/train/
data/splits/test/
```

Create edit/eval sets:

```bash
python clustering/cluster_train.py \
  --train_dir data/splits/train \
  --test_dir data/splits/test \
  --num_clusters "$k" \
  --sample_size "$s"
```

Output:

```text
data/edit_sets/k${k}_s${s}/
data/eval_sets/test/k${k}_s${s}/
```

## FERMI

Run FERMI on the clustered split produced by `clustering/cluster_train.py`:

```bash
python fermi/run_fermi.py \
  --num_clusters "$k" \
  --sample_size "$s"
```

By default this reads:

```text
data/edit_sets/k${k}_s${s}/
data/eval_sets/test/k${k}_s${s}/
```

and writes:

```text
results/fermi/k${k}_s${s}/
  <respondent_id>/
    optimization_log.json
    test_inference_results.json
```

Evaluate FERMI predictions by parsing each prediction into one of the prompt
options and comparing it with `target`:

```bash
python evaluation/evaluate_fermi.py --config-name k${k}_s${s}
```

Output:

```text
results/fermi_evaluation/k${k}_s${s}/
  summary.json
  <respondent_id>_eval.json
```

## Editing

Clustered smoke editing, one edit per respondent:

```bash
python scripts/run_editing.py \
  --ds_folder data/edit_sets/k${k}_s${s} \
  --alg_name AlphaEdit \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --hparams_fname Llama3.1-8B.json \
  --ds_name opinionqa \
  --skip_generation_tests \
  --conserve_memory \
  --single_gpu_load
```

Notes:

- `--single_gpu_load` loads the whole model onto `cuda:0`; omit it or expose
  multiple GPUs if the selected GPU is too small.
- Remove `--skip_generation_tests` if you need fluency/generation metrics such
  as `post_ngram_entropy`.
- Add `--downstream_eval_steps 1` if you need GLUE output files.

Editing outputs:

```text
results/editing/AlphaEdit/run_XXX/
```

## Evaluation

General prompt:

```bash
python evaluation/evaluate.py \
  --eval-root data/eval_sets \
  --eval-config k${k}_s${s} \
  --weights-root results/editing/AlphaEdit/run_XXX \
  --prompt-template data/prompts/survey.txt \
  --history-mode none \
  --output-dir results/evaluation/k${k}_s${s}_run_XXX \
  --batch-size 4 \
```
Evaluation outputs:

```text
results/evaluation/<run_name>/
  summary.json
  edited_train/...
  test/...
```

## Editing Summary

Print editing metrics:

```bash
python scripts/summarize_editing.py --dir_name AlphaEdit --runs run_XXX --no-output
```

# Acknowledgment
Our code for editing is based on [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit)
