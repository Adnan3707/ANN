# FashionFinder: product-image classification project

This repository is a realistic junior machine-learning-engineer assignment. You work for **FashionFinder**, an online clothing marketplace. The Catalog Operations team wants an automated system that routes supplier product images to the correct top-level category before a human reviewer checks them.

The supplied training proxy is the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset: 28 × 28 grayscale product images across ten categories. Your job is to create a reproducible Keras/TensorFlow training pipeline and recommend whether its results are good enough for a controlled pilot.

## What you are expected to deliver

1. A baseline fully connected neural network and a CNN experiment.
2. Saved model and metrics for each run in `artifacts/<run-name>/`.
3. A short decision note comparing the models and describing failure modes.
4. A final recommendation against the acceptance criteria in [JOB_BRIEF.md](JOB_BRIEF.md).

The repository gives you a working baseline pipeline; the tasks marked **Your work** are deliberately yours to implement, evaluate, and explain.

## Quick start

Use Python 3.10–3.12. Creating a virtual environment is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train a 5-epoch fully connected baseline.
python -m src.train --model baseline --epochs 5 --run-name baseline-v1

# Train the convolutional candidate model.
python -m src.train --model cnn --epochs 12 --run-name cnn-v1

# Run the included smoke tests.
pytest -q
```

Fashion-MNIST downloads automatically to `data/` on the first run. Training output is stored under `artifacts/`, which is ignored by Git.

## Repository map

```text
JOB_BRIEF.md        Business context, constraints, tasks, and acceptance criteria
src/data.py         Downloading, deterministic data split, preprocessing, tf.data pipeline
src/models.py       Baseline and CNN model factories
src/train.py        Experiment CLI, callbacks, metrics, artifact persistence
tests/              Fast checks for data and model contracts
```

## Suggested working sequence

1. Read [JOB_BRIEF.md](JOB_BRIEF.md) as if it came from your manager.
2. Run `baseline-v1`, inspect `metrics.json` and `confusion_matrix.csv`.
3. Run `cnn-v1` with the same data split. Compare validation and test performance.
4. Complete the **Your work** items in the brief. Make one change at a time and name runs clearly.
5. Write your decision note in `reports/model_recommendation.md` (a template is included).

## Useful commands

```powershell
# Small, fast smoke-training run (downloads the dataset if needed).
python -m src.train --model cnn --epochs 1 --limit-train 1024 --limit-test 256 --run-name smoke

# See all options.
python -m src.train --help
```

## Notes on responsible use

Fashion-MNIST is a benchmark, not a production catalog. A real deployment would require supplier-specific images, representative labels, monitoring for data drift, a human-review path, privacy review, and bias/quality assessment by category and seller segment. This exercise uses it only to practise the engineering workflow.
