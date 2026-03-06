# Autextification Replication

Replication and extension of **"I've Seen Things You Machines Wouldn't Believe: Measuring Content Predictability to Identify Automatically-Generated Text"** by Przybyła et al. (2023).

Hybrid neural-linguistic models for machine-generated text detection that combine token-level predictability features from multiple causal LMs with fine-tuned encoder backbones (RoBERTa, DeBERTa, XLM-R, ModernBERT). Extends the original work with multilingual probabilistic models (Qwen, LLaMA, XGLM, BLOOM), style-based linguistic features, and SHAP-based interpretability.

**Paper**: [CEUR-WS.org/Vol-3496/autextification-paper7.pdf](https://ceur-ws.org/Vol-3496/autextification-paper7.pdf)
**Dataset**: [symanto/autextification2023](https://huggingface.co/datasets/symanto/autextification2023)

## Setup

### Installation

```bash
git clone https://github.com/<your-username>/autextification-replication.git
cd autextification-replication

conda create -n replication_study python=3.11 -y
conda activate replication_study

pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

> **Note**: Exact library versions in `requirements.txt` are pinned for reproducibility. Results may vary with different `transformers` or `torch` versions.

### Data

Download the AuTexTification 2023 dataset from Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("symanto/autextification2023")
```

Or download manually from [symanto/autextification2023](https://huggingface.co/datasets/symanto/autextification2023).

Place the data in `data/data/train/` and `data/data/test/` following the expected structure:

```
data/
├── data/
│   ├── train/{subtask_1,subtask_2}/{en,es}/
│   └── test/{subtask_1,subtask_2}/{en,es}/
├── features/   # cached .npy feature matrices (generated at runtime)
└── out/        # checkpoints, results, SHAP artifacts
```

### Resources

External resources required by the word frequency and grammar features (from the original authors).

**Word frequency matrices** (Google Book Corpus, preprocessed):

- English: [Google Drive](https://drive.google.com/file/d/1PwcEHgR8jU3M9_2QHCW0_NO0rrc2kRw1/view?usp=sharing)
- Spanish: [Google Drive](https://drive.google.com/file/d/1jsFPoYlCf9U8BfKBASnrs-OtNEqySnft/view?usp=share_link)

Place each `.tsv.gz` file in the corresponding subfolder:

```
resources/
├── en/    # English frequency matrix
└── es/    # Spanish frequency matrix
```

Alternatively, run `feature_extraction/download_ngram.py` to download and process the raw n-gram data (slow).

**Grammar checking** (LanguageTool local server on port 8010):

```bash
docker pull erikvl87/languagetool
docker run --rm -p 8010:8010 erikvl87/languagetool
```

The grammar feature extractor uses `language-tool-python` to query this local API.

## Model Variants

| Variant | Architecture | Description |
|---|---|---|
| `pred` | BiLSTM | LSTM on probabilistic features only (no encoder) |
| `flm` | Encoder | Fine-tuned language model (RoBERTa / DeBERTa / etc.) |
| `pred_flm` | BiLSTM + Encoder | Hybrid: BiLSTM on prob features fused with encoder |
| `pred_flm_add` | BiLSTM + Encoder + RF features | Hybrid+: adds word frequency and grammar features |

## Scripts

### Training

- **`scripts/training.py`** — Train a single model variant with a chosen encoder and seed.

```bash
python scripts/training.py --subtask subtask_1 --lang en --model_variant pred_flm --config baseline --seed 10
python scripts/training.py --subtask subtask_2 --lang en --model_variant flm --config multilingual
```

- **`scripts/sweeper.py`** — Sweep over multiple encoder/variant combinations and aggregate results to a TSV.

```bash
python scripts/sweeper.py --subtask all --seeds 10 42 123
```

- **`scripts/run_seeds.py`** — Run `training.py` across multiple seeds and report mean +/- std F1.

```bash
python scripts/run_seeds.py --subtask subtask_1 --lang en --model_variant pred_flm --config baseline --seeds 10 42 123
```

### Multilingual Experiments

- **`scripts/run_experiments.py`** — Experiment pipeline with configurable probabilistic model presets (XGLM-focused, mGPT-focused, large multilingual). Shares prob features across variants for efficiency.

```bash
python scripts/run_experiments.py --experiment multilingual_xglm --variant pred_flm --subtask subtask_1 --lang en
python scripts/run_experiments.py --experiment all --variant all --subtask all --lang all
```

### UltraHybrid (Two-Stage Pipeline)

- **`scripts/training_ultrahybrid.py`** — Stage 1 trains a Hybrid+ neural model, Stage 2 feeds its output probabilities together with linguistic features into RF / XGBoost / MLP classifiers.

```bash
python scripts/training_ultrahybrid.py --subtask subtask_1 --lang en --config baseline
python scripts/training_ultrahybrid.py --subtask subtask_1 --lang en --config baseline --style
```

### Features-Only Classifiers

- **`scripts/training_features.py`** — Aggregates token-level features (probabilistic, word frequency, grammar) to document-level, combines with linguistic (and optional style) features, and trains RF / XGBoost / MLP directly (no neural model).

```bash
python scripts/training_features.py --subtask subtask_1 --lang en --config baseline
python scripts/training_features.py --subtask subtask_1 --lang en --config baseline --style
```

### Linguistic Feature Classifiers

- **`scripts/training_lingrf.py`** — LingRF and LingRF+PredOut models combining linguistic features, style features, and Random Forest with optional SHAP analysis.

```bash
python scripts/training_lingrf.py --subtask subtask_1 --lang en --variant lingrf_style
```

### Analysis & Utilities

- **`scripts/error_analysis.py`** — Run inference on a saved checkpoint and export confusion matrices, per-class metrics, and misclassified samples.
- **`scripts/plot_shap.py`** — Generate SHAP bar/beeswarm plots from saved `.npz` files.
- **`scripts/extract_style_features.py`** — Precompute and cache style-feature matrices.
- **`scripts/tune_optuna.py`** — Optuna hyperparameter tuning for aggregated-feature classifiers (RF / XGBoost / MLP).
- **`scripts/compare_features.py`** — Bit-for-bit verification of feature extractor equivalence.
- **`scripts/data_split_LDA.py`** — LDA-based topic-balanced fold generation.

## Project Structure

```
autextification-replication/
├── scripts/
│   ├── training.py              # Single model training
│   ├── training_ultrahybrid.py  # Two-stage: Hybrid+ → RF / XGB / MLP
│   ├── training_features.py     # Features-only: aggregated features → RF / XGB / MLP
│   ├── training_lingrf.py       # LingRF + style features + SHAP
│   ├── sweeper.py               # Encoder/variant sweep
│   ├── run_seeds.py             # Multi-seed aggregation
│   ├── run_experiments.py       # Multilingual experiment pipeline
│   ├── error_analysis.py        # Post-hoc error analysis
│   ├── plot_shap.py             # SHAP visualizations
│   ├── extract_style_features.py # Style feature caching
│   ├── tune_optuna.py           # Optuna hyperparameter search
│   ├── compare_features.py      # Feature extractor verification
│   └── data_split_LDA.py        # LDA-based data splitting
│
├── feature_extraction/
│   ├── feature_generator.py     # Abstract base class
│   ├── probabilistic_features.py # LM-based per-token features (GPT-2, Qwen, LLaMA, etc.)
│   ├── grammar_features.py      # Grammar checking + word frequency
│   ├── linguistic_features.py   # POS, dependency, NER features + LingRF classifiers
│   ├── style_features.py        # Stylometric features (TTR, sentence stats, etc.)
│   └── download_ngram.py        # N-gram data downloader
│
├── models/
│   └── hybrid.py                # PredLSTM, FLMRoBERTa, HybridBiLSTMRoBERTa
│
├── utils/
│   ├── constants.py             # Shared constants, paths, encoder maps
│   ├── data_utils.py            # Data loading and splitting
│   ├── feature_utils.py         # Feature computation orchestration
│   ├── train_utils.py           # Training / evaluation loops
│   ├── training_pipeline.py     # Model construction, tokenization, train-evaluate
│   ├── classifier_utils.py      # Stage-2 classifiers: RF, XGBoost, PyTorch MLP
│   └── logging_utils.py         # Tee stdout logger
│
│
├── EDA.ipynb                    # Exploratory data analysis
├── requirements.txt
└── README.md
```

## Citation

Our paper:

TBA

Original paper:

```bibtex
@inproceedings{przybyla2023seen,
    title = "I've Seen Things You Machines Wouldn't Believe: Measuring Content Predictability to Identify Automatically-Generated Text",
    author = "Przybyła, Piotr and Duran-Silva, Nicolau and Egea-Gómez, Santiago",
    booktitle = "Procesamiento del Lenguaje Natural",
    year = "2023",
    address = "Jaén, Spain"
}
```

Dataset:

```bibtex
@inproceedings{autextification2023,
    title = "Overview of AuTexTification at IberLEF 2023: Detection and Attribution of Machine-Generated Text in Multiple Domains",
    author = "Sarvazyan, Areg Mikael and González, José Ángel and Franco-Salvador, Marc and Rangel, Francisco and Chulvi, Berta and Rosso, Paolo",
    booktitle = "Procesamiento del Lenguaje Natural",
    year = "2023"
}
```
