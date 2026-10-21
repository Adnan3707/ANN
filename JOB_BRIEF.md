# Job brief: FashionFinder catalog-routing pilot

**Role:** Junior Machine Learning Engineer  
**Manager:** ML Platform Lead  
**Timeline:** 5 working days  
**Status:** Starter assignment

## Business situation

FashionFinder receives thousands of supplier images every week. Analysts currently select a product category manually, delaying product publication. We want a classifier that proposes one of ten high-level categories and sends uncertain or error-prone cases to a reviewer.

For this practice project, Fashion-MNIST represents the historical labelled image archive.

| Dataset label | Business-facing category |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Objective

Build and assess a reproducible image-classification service candidate using TensorFlow/Keras. It must improve on a dense-neural-network baseline and provide enough evidence for the Operations team to decide whether to pilot it with human review.

## Constraints

- Use Python, TensorFlow, and Keras.
- Use only Fashion-MNIST for modelling in this assignment.
- Keep the official test set untouched until choosing a candidate model.
- Make runs reproducible: record seed, parameters, validation/test metrics, and training history.
- The system is a decision-support tool. It must not silently auto-route predictions below a confidence threshold.
- Aim for CPU-friendly training; document any GPU use.

## Acceptance criteria

| Requirement | Pilot threshold |
| --- | --- |
| Overall test accuracy | ≥ 88% |
| CNN improvement over baseline | ≥ 2 percentage points on validation accuracy |
| Worst category recall | ≥ 75% |
| Low-confidence routing | Predictions below a chosen threshold are marked for human review |
| Reproducibility | A fresh run produces metrics/artifacts with the recorded arguments |
| Explainability | Report the three most frequent category confusions and a plausible business impact |

## Definition of done

1. `python -m src.train --model baseline ...` produces a saved baseline and metrics.
2. `python -m src.train --model cnn ...` produces a candidate model and metrics.
3. You compare both runs fairly: same split, seed, preprocessing, and reporting method.
4. `reports/model_recommendation.md` states **pilot**, **pilot with conditions**, or **do not pilot**, citing evidence.
5. The recommendation identifies the confidence threshold and reviewer handling for low-confidence predictions.

## Your work

The starter code intentionally stops short of a finished business decision. Do these tasks as separate, named experiments.

### Core tasks

- [ ] Run the dense baseline and record its validation and test accuracy.
- [ ] Run the CNN and compare it to the baseline using `metrics.json`.
- [ ] Find the three largest off-diagonal counts in `confusion_matrix.csv`. Explain why these errors are risky or acceptable to the catalog team.
- [ ] Select a confidence threshold (for example, 0.70–0.90). Calculate the review rate and accuracy among auto-routed images.
- [ ] Update the report template with your recommendation.

### Engineering tasks

- [ ] Add a `--confidence-threshold` command-line option. Save `review_rate`, `auto_route_rate`, and `auto_route_accuracy` in `metrics.json`.
- [ ] Add a unit test covering the new routing calculation.
- [ ] Add an augmentation experiment (`RandomFlip`, `RandomRotation`, or `RandomTranslation`) and justify whether it helps catalogue images.
- [ ] Add a small hyperparameter experiment table to the report: model, learning rate, batch size, epochs, validation accuracy, test accuracy, and notes.

### Stretch tasks

- [ ] Build a small `src/predict.py` command that loads a `.keras` model and predicts an image file.
- [ ] Add per-class precision, recall, and F1 metrics without introducing data leakage.
- [ ] Save an error gallery of incorrectly classified test images with true/predicted/confidence labels.
- [ ] Add a simple data-drift check for image brightness or pixel distribution between a reference and incoming batch.

## Questions your manager will ask

Prepare answers grounded in your experiments:

1. Which categories are confused most often, and why might their images look similar at 28 × 28 pixels?
2. Is overall accuracy hiding a weak category?
3. At your selected confidence threshold, what share of products still need human review?
4. What could fail when moving from Fashion-MNIST to supplier photos?
5. What will you monitor after launch?

