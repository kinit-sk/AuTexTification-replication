# AuTexTification Replication

Replication/extension code for machine-generated text detection with
predictability features, encoder models, stylometric features, LingRF,
ablations, SHAP, Optuna tuning, and MULTITuDE mAA experiments.

Run everything from this directory:

```bash
cd autextification--replication
```

Use file execution for most scripts, for example `python scripts/training.py`.
MULTITuDE also supports `python -m scripts.multitude`.

## Setup

```bash
conda create -n replication_study python=3.11 -y
conda activate replication_study

pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

Some multilingual models are large or gated, so log in to Hugging Face when
needed:

```bash
huggingface-cli login
```

## Data

AuTexTification files:

```text
data/data/train/subtask_1/en/train.tsv
data/data/train/subtask_1/es/train.tsv
data/data/train/subtask_2/en/train.tsv
data/data/train/subtask_2/es/train.tsv
data/data/test/subtask_1/en/test.tsv
data/data/test/subtask_1/es/test.tsv
data/data/test/subtask_2/en/test.tsv
data/data/test/subtask_2/es/test.tsv
```

TSV columns: `text`, `label`. Labels: `human/generated` for `subtask_1`, `A-F`
for `subtask_2`.

Word-frequency resources:

```text
resources/en/word_freq_matrix.tsv.gz
resources/es/word_freq_matrix.tsv.gz
```

Raw n-gram download/processing, if you do not already have the resources:

```bash
python feature_extraction/download_ngram.py
```

Grammar features use LanguageTool. Local server:

```bash
docker pull erikvl87/languagetool
docker run --rm -p 8010:8010 erikvl87/languagetool
```

MULTITuDE default dataset: `data/multitude_v3_mAA.csv.gz`.
Required columns: `text`, `multi_label`, `split`, `language`, `source`.

## Structure

```text
feature_extraction/   probability, grammar, linguistic, style features
models/               neural models
scripts/              runnable experiments
scripts/ablation/     style ablation and permutation importance
scripts/multitude/    MULTITuDE mAA package
utils/                data, constants, training, classifiers, logging
```

Main outputs:

```text
logs/
data/features/
data/out/checkpoints/
data/out/results/
data/out/shap/
data/out/shap/plots/
data/out/tuning/
data/out/ablation_probs/
data/out/perm_importance/
```

## Variants

Neural:

```text
pred          BiLSTM on predictability features only
flm           encoder-only classifier
pred_flm      predictability BiLSTM + encoder
pred_flm_add  pred_flm + word-frequency + grammar token features
```

LingRF:

```text
lingrf                 linguistic features -> RF
lingrf_style           linguistic + style -> RF
lingrf_predout         linguistic + BiLSTM output probabilities -> RF
lingrf_style_predout   linguistic + style + BiLSTM output probabilities -> RF
```

Configs:

```text
baseline:
  en -> roberta-base
  es -> bertin-project/bertin-roberta-base-spanish
multilingual:
  encoder -> microsoft/mdeberta-v3-base
```

## Run Reference

### Neural Training

- Single neural run:

```bash
python scripts/training.py --subtask subtask_1 --lang en --model_variant pred_flm --config baseline --seed 10
```

Args: `--subtask subtask_1|subtask_2`, `--lang en|es`,
`--model_variant pred|flm|pred_flm|pred_flm_add`,
`--config baseline|multilingual`, `--encoder_id`, `--seed`,
`--legacy_roberta_compat`.

- Multi-seed subprocess grid:

```bash
python scripts/run_seeds.py --subtask subtask_1 --lang en --model_variant pred_flm --config baseline --seeds 10,42,123
```

Args: `--subtask`, `--lang`, `--model_variant`, `--config` accept one value,
`all`, or comma lists. `--seeds` is comma-separated.

- Fixed in-process sweep:

```bash
python scripts/sweeper.py --subtask all --seeds 10 42 123
```

Args: `--subtask subtask_1|subtask_2|all`, `--seeds`, `--epochs`,
`--freeze_epochs`, `--batch_size`, `--num_workers`.

- Multilingual probability presets:

```bash
python scripts/run_experiments.py --experiment multilingual_xglm --variant pred_flm --subtask subtask_1 --lang en
```

Args: `--experiment multilingual_xglm|multilingual_mgpt|multilingual_large|all|comma-list`,
`--variant pred|flm|pred_flm|pred_flm_add|all`, `--subtask`, `--lang`,
`--batch-size`. Here, `--variant all` means `pred_flm` and `pred_flm_add`.

### Feature-Only, UltraHybrid, Optuna

- Aggregated token features + linguistic/style -> RF/XGB/MLP:

```bash
python scripts/training_features.py --subtask subtask_1 --lang en --config baseline --seed 10
python scripts/training_features.py --subtask subtask_1 --lang en --config baseline --seed 10 --style
```

- Hybrid+ probabilities + linguistic/style -> RF/XGB/MLP:

```bash
python scripts/training_ultrahybrid.py --subtask subtask_1 --lang en --config baseline --seed 10
python scripts/training_ultrahybrid.py --subtask subtask_1 --lang en --config baseline --seed 10 --style
```

Both support `--subtask subtask_1|subtask_2`, `--lang en|es`,
`--config baseline|multilingual`, `--seed`, `--style`. They save NPZ files to
`data/out/tuning/`.

- Tune saved NPZ datasets:

```bash
python scripts/tune_optuna.py --npz data/out/tuning/subtask_1_en_baseline_features_stage2_data.npz --models rf xgb mlp --trials 60
python scripts/tune_optuna.py --source agg --models rf xgb --trials 100
python scripts/tune_optuna.py --source ultrahybrid --models mlp --trials 80 --save_best
python scripts/tune_optuna.py --source lingrf --models rf --trials 100
```

Args: `--npz`, `--trials`, `--models rf|xgb|mlp`, `--seed`, `--device`,
`--save_best`, `--source agg|ultrahybrid|lingrf`.

### LingRF

- Train LingRF variants:

```bash
python scripts/training_lingrf.py --subtask subtask_1 --lang en --variant lingrf_style --no-shap
```

Args: `--subtask subtask_1|subtask_2` or omitted for both, `--lang en|es` or
omitted for both, `--variant lingrf|lingrf_style|lingrf_predout|lingrf_style_predout|all`,
`--multilingual`, `--no-shap`, `--shap-samples`.

- Fixed LingRF scenario matrix:

```bash
python scripts/run_lingrf.py --subtask all --lang all
python scripts/run_lingrf.py --subtask subtask_1 --lang en --shap
```

Args: `--subtask subtask_1|subtask_2|all`, `--lang en|es|all`, `--shap`.

- Precompute style features:

```bash
python scripts/extract_style_features.py --subtask subtask_1 --lang en
python scripts/extract_style_features.py --subtask all --lang all
```

Args: `--subtask subtask_1|subtask_2|all`, `--lang en|es|all`.

### Ablations

- Precompute PredOut probabilities:

```bash
python scripts/ablation/precompute_lstm_probs.py
python scripts/ablation/precompute_lstm_probs.py --subtask subtask_1 --lang en
python scripts/ablation/precompute_lstm_probs.py --subtask subtask_1 --lang en --multilingual --force
```

Args: `--subtask`, `--lang`, `--multilingual`, `--force`. Omit `--subtask` or
`--lang` to run both.

- One ablation config:

```bash
python scripts/ablation/training_ablation.py --subtask subtask_1 --lang en --variant lingrf_style --exclude-style-groups LexicalDiversity --seed 10 --no-shap
```

Args: `--subtask subtask_1|subtask_2`, `--lang en|es`,
`--variant lingrf_style|lingrf_style_predout`, `--exclude-style-groups`,
`--multilingual`, `--seed`, `--no-shap`, `--shap-samples`.

Style groups: `LexicalDiversity`, `SentenceStructure`, `RepetitionPatterns`,
`WordLevelStatistics`, `FunctionalStylisticMarkers`, `ReadabilityMetrics`,
`PunctuationUsage`.

- Full ablation runner:

```bash
python scripts/ablation/run_ablation.py --subtask all --lang all --variants lingrf_style --seeds 10
python scripts/ablation/run_ablation.py --variants lingrf_style_predout --multilingual --seeds 10
python scripts/ablation/run_ablation.py --paper-all --seeds 10 11 12
```

Args: `--subtask`, `--lang`, `--multilingual`, `--variants`, `--seeds`,
`--shap`, `--paper-all`, `--paper-all-shap`, `--paper-shap-samples`.

- Grouped permutation importance:

```bash
python scripts/ablation/permutation_importance.py
python scripts/ablation/permutation_importance.py --subtask subtask_1 --lang en --variants lingrf_style --seeds 10 11 12 --repeats 5
python scripts/ablation/permutation_importance.py --variants lingrf_style_predout --multilingual
```

Args: `--subtask`, `--lang`, `--variants`, `--seeds`, `--repeats`,
`--multilingual`.

### SHAP And Error Analysis

- Plot SHAP files from `data/out/shap/`:

```bash
python scripts/plot_shap.py --subtask subtask_1 --lang en --variant lingrf_style
python scripts/plot_shap.py --all --mode both --top-n 20
```

Args: `--subtask`, `--lang`, `--variant`, `--all`,
`--mode standard|multiclass|both`, `--top-n`, `--grouped`, `--no-grouped`,
`--no-feature-level`, `--direction-summary`, `--no-direction-summary`.

- Error analysis from neural checkpoint:

```bash
python scripts/error_analysis.py --checkpoint_path data/out/checkpoints/subtask_1_en_flm_baseline_seed10_epoch3.pt --split test
```

Args: `--checkpoint_path`, `--subtask`, `--lang`, `--model_variant`,
`--config`, `--seed`, `--encoder_id`, `--split train|dev|test`, `--output_dir`,
`--compare_predictions`.

### MULTITuDE

```bash
python -m scripts.multitude --validate-only
python -m scripts.multitude --variant all
python scripts/multitude/run.py --variant hybrid_multilingual
python scripts/multitude/run.py --variant lingrf_predout_multilingual
```

Args: `--dataset`, `--variant all|hybrid_multilingual|lingrf_predout_multilingual`,
`--seed`, `--epochs`, `--freeze-epochs`, `--lstm-epochs`, `--batch-size`,
`--prob-batch-size`, `--rf-estimators`, `--rf-max-depth`, `--encoder-id`,
`--prob-models`, `--force-recompute-prob`, `--validate-only`.

## Notes

- Neural CLIs use a random 80/20 train/dev split from `train.tsv`.
- Default seed is usually `10`; MULTITuDE defaults to `42`.
- Probability feature extraction is expensive and is recomputed by most neural
  scripts.
- No shell runners are present in the current `scripts/` directory.

## Citation

Our replication paper (accepted to Findings of EMNLP 2026):

```bibtex
@misc{skurla2026interpretablepredictabilitybasedaitext,
      title={Interpretable Predictability-Based AI Text Detection: A Replication Study}, 
      author={Adam Skurla and Dominik Macko and Jakub Simko},
      year={2026},
      eprint={2603.15034},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.15034}, 
}
```

Original paper:

```bibtex
@inproceedings{przybyla2023seen,
  title = {I've Seen Things You Machines Wouldn't Believe: Measuring Content Predictability to Identify Automatically-Generated Text},
  author = {Przybyla, Piotr and Duran-Silva, Nicolau and Egea-Gomez, Santiago},
  booktitle = {Procesamiento del Lenguaje Natural},
  year = {2023},
  address = {Jaen, Spain}
}
```

Dataset:

```bibtex
@inproceedings{autextification2023,
  title = {Overview of AuTexTification at IberLEF 2023: Detection and Attribution of Machine-Generated Text in Multiple Domains},
  author = {Sarvazyan, Areg Mikael and Gonzalez, Jose Angel and Franco-Salvador, Marc and Rangel, Francisco and Chulvi, Berta and Rosso, Paolo},
  booktitle = {Procesamiento del Lenguaje Natural},
  year = {2023}
}
```
