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
python scripts/run_clustering.py \
  --train_dir data/splits/train \
  --test_dir data/splits/test \
  --num_clusters "$k" \
  --sample_size "$s" \
```

Output:

```text
data/edit_sets/k${k}_s${s}/
data/eval_sets/test/k${k}_s${s}/
```

## FERMI

Run FERMI on the clustered split produced by `scripts/run_clustering.py`:

```bash
python scripts/run_fermi.py \
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
python scripts/run_fermi_evaluation.py --config-name k${k}_s${s}
```

Output:

```text
results/fermi_evaluation/k${k}_s${s}/
  summary.json
  <respondent_id>_eval.json
```

## Experiment Consistency Analysis

Compare two evaluated experiment directories and correlate each respondent's
target-answer cluster consistency with the accuracy difference between the
experiments:

```bash
python scripts/run_cluster_consistency.py \
  --experiment-a results/evaluation/experiment_a/test/k18_s200 \
  --experiment-b results/evaluation/experiment_b/test/k18_s200 \
  --name-a experiment_a \
  --name-b experiment_b \
  --output-dir results/analysis/experiment_a_vs_b_test
```

The analysis pairs matching respondent/question IDs and defines improvement as:

```text
accuracy_improvement = accuracy_experiment_b - accuracy_experiment_a
```

Both result directories should contain evaluated entries with `cluster_id`,
`target`, and `is_correct`. If `is_correct` is absent, pass `--answer-key-a`
and/or `--answer-key-b`.

Output:

```text
results/analysis/experiment_a_vs_b_test/
  respondent_cluster_consistency.csv
  respondent_consistency_accuracy.csv
  summary.json
```


### Human-Labeled Category Split

This split is based on human-labeled semantic topic/category annotations, not
hidden-state clustering.

```text
data/edit_sets/cat120_120/
data/eval_sets/same_topic/cat120_120/
data/eval_sets/other_topics/cat120_120/
```

Respondents are anonymized as `c120_001` through `c120_010`. Each edit set has
120 edit examples.

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
python scripts/run_evaluation.py \
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

## Ordinal Metrics

Compute ordinal MMAE and CEM:

```bash
python scripts/run_ordinal_metrics.py \
  --input results/evaluation/k${k}_s${s}_run_XXX \
  --distribution-dir data/OpinionQA/distribution \
  --output-dir results/evaluation_metrics/k${k}_s${s}_run_XXX
```

Outputs:

```text
ordinal_metrics.csv
ordinal_summary.json
```

`data/OpinionQA/distribution/` contains the required
`<wave>_default_human.csv` files for CEM.

## Editing Summary

Print editing metrics:

```bash
python scripts/summarize_editing.py --dir_name AlphaEdit --runs run_XXX --no-output
```

# Acknowledgment
Our code for editing is based on [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit)
