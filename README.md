# visual-question-answering

## Problem Statement
Visual Question Answering (VQA) aims to answer natural language questions about
images. This repository focuses on open-ended VQA for natural images with the
goal of improving accuracy on diverse question types by combining vision models
and large language models (LLMs).

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Setup
Run the dataset download script to fetch VQA v2.0:
```bash
python -m vqa.data --download
```
A sample visualization of an image, question, and answer can be produced with:
```bash
python -m vqa.data --show-sample
```

## Quick Start
Train the baseline model on VQA v2.0:
```bash
python -m vqa.train --epochs 1
```
Evaluate on the validation split:
```bash
python -m vqa.evaluate
```

## Ethical Considerations
VQA datasets can contain societal biases (e.g., gender or cultural stereotypes).
We recommend examining dataset statistics and being cautious when deploying
models trained on this data.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Related Work
A short list of recent papers can be found in [RELATED_WORK.md](RELATED_WORK.md).
