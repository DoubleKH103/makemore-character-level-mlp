# Makemore Part 3 — Character-Level MLP

A from-scratch character-level language model implemented in PyTorch. The project builds a deeper MLP language model with learned character embeddings, manually implemented BatchNorm, training diagnostics, evaluation, and autoregressive text generation.

## Project goal

Learn the probability of the next character given a fixed context:

```text
context → embeddings → MLP → logits → probabilities → next character
```

With a context size of 3, the dataset contains examples such as:

```text
... → h
..h → o
.ho → m
hom → e
ome → .
```

## What I implemented

- Character vocabulary with `stoi` / `itos` mappings
- Context-window dataset construction
- Train / validation / test splits
- Learned character embeddings
- Multi-layer perceptron
- Custom `Linear` layer
- Custom `BatchNorm1d` layer
- Custom `Tanh` activation
- Cross-entropy loss
- Mini-batch gradient descent
- Learning-rate decay
- Activation diagnostics
- Gradient diagnostics
- Parameter-update diagnostics
- Train / validation / test evaluation
- Autoregressive character sampling

## Architecture

```text
Character IDs
      ↓
Embedding
      ↓
Flattened Context
      ↓
Linear → BatchNorm → Tanh
      ↓
Linear → BatchNorm → Tanh
      ↓
Linear → BatchNorm → Tanh
      ↓
Linear → BatchNorm → Tanh
      ↓
Linear → BatchNorm → Tanh
      ↓
Linear → BatchNorm
      ↓
Output logits
```

## Repository structure

```text
makemore-character-level-mlp/
├── data/
│   └── .gitkeep
├── notebooks/
│   └── makemore_p3.ipynb
├── scripts/
│   └── download_data.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

```bash
git clone <your-repository-url>
cd makemore-character-level-mlp
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download the dataset

```bash
python scripts/download_data.py
```

Then open `notebooks/makemore_p3.ipynb` in Jupyter or VS Code and run the notebook from top to bottom.

## Dataset

The project uses the `names.txt` dataset from Andrej Karpathy's `makemore` repository. The dataset is downloaded into `data/names.txt` and is intentionally ignored by Git rather than committed to the repository.

## Training diagnostics

The notebook does more than track loss. It inspects:

- Activation distributions
- `tanh` saturation
- Gradient distributions
- Gradient-to-weight ratios
- Parameter update magnitude
- Train / validation / test loss

These diagnostics make it possible to reason about initialization, normalization, optimization, and generalization instead of treating the model as a black box.

## Key learning outcome

The main progression in this project is from simply training a model to understanding what is happening inside it:

```text
Build → Train → Inspect activations → Inspect gradients
      → Evaluate generalization → Generate text
```

## Roadmap

This project is part of a progression toward modern neural language models:

```text
Bigram Model
    ↓
Character-Level MLP
    ↓
Deeper MLP + BatchNorm
    ↓
Attention
    ↓
Self-Attention
    ↓
Transformer
```

## Reference

Inspired by Andrej Karpathy's `makemore` educational series.
