# Model recommendation — FashionFinder

**Author:**  
**Date:** 2026-10-21  
**Decision:** Pilot with conditions

## Evidence

| Run | Model | Key settings | Validation accuracy | Test accuracy | Worst-category recall |
| --- | --- | --- | ---: | ---: | ---: |
| baseline-v1 | Dense network (ANN) | lr 0.001, batch 128, 5 epochs | 88.06% | 86.66% | 66.40% (Shirt) |
| cnn-v1 | CNN | lr 0.001, batch 128, 12 epochs | 92.96% | 92.31% | 78.70% |
| cnn-aug-v1 | CNN + random translation | lr 0.001, batch 128, 12 epochs | 90.94% | 90.81% | 63.90% |

The CNN improved validation accuracy by 4.90 percentage points over the baseline, exceeding the 2-point target. Its test accuracy exceeds the 88% target and its worst-category recall exceeds the 75% target. Translation augmentation did not help in this experiment: the augmented model was lower by 2.02 points in validation accuracy and 1.50 points in test accuracy than cnn-v1, and missed the worst-category recall target.

The baseline was trained for 5 epochs and the CNN for 12, so this comparison did not use a matched epoch budget. The reported results are useful experiment evidence, but test metrics were inspected across multiple experiments; they should not be treated as an untouched final evaluation for deployment.

## Error analysis

The three largest off-diagonal counts for cnn-v1 were:

1. **T-shirt/top → Shirt (89 images):** These categories have similar silhouettes at 28 × 28 pixels. A wrong category can misfile a product and make it harder for customers to find.
2. **Shirt → T-shirt/top (87 images):** This is the reverse confusion and may similarly affect category filters and product discovery.
3. **Coat → Shirt (60 images):** Confusing outerwear with a shirt is a larger category mismatch and could mislead shoppers or require catalog correction.

These counts describe this test set, not expected rates for supplier photos. Human review and validation on representative marketplace images are needed before relying on these predictions.

## Confidence routing proposal

**Threshold:** 0.80  
**Auto-route rate:** 87.73%  
**Human-review rate:** 12.27%  
**Accuracy of auto-routed predictions:** 97.08%

At this threshold, about one in eight benchmark images is flagged for review. Of the remaining auto-routed images, 97.08% were correct on the measured test set. This is a promising workload/error trade-off for investigation, not a production guarantee: the threshold should be assessed and calibrated using a separate, representative supplier-image validation set, and low-confidence images must continue to go to a human reviewer.

## Recommendation and next steps

**Recommendation: Pilot with conditions.** The original CNN meets the benchmark test-accuracy threshold (92.31% versus 88%), the CNN validation improvement threshold (+4.90 percentage points versus +2), and the worst-category recall threshold (78.70% versus 75%). The 0.80 routing experiment also demonstrates a human-review path for lower-confidence predictions.

Do not deploy this benchmark model directly to automatically route real catalog products. Fashion-MNIST images are small, grayscale, and centered; supplier photos can vary in backgrounds, lighting, resolution, pose, cropping, and product presentation. Before a controlled pilot:

- Collect and label a representative sample of supplier images, keeping suppliers and near-duplicate products separated between training, validation, and evaluation sets.
- Evaluate the model and select the confidence threshold on that data before measuring final performance on a held-out evaluation set. Keep all below-threshold and otherwise flagged cases in human review.
- During the pilot, monitor overall and per-category precision/recall, confidence and review rates, correction rates from reviewers, and changes in image brightness or other input distributions. Review results by supplier and category, and pause auto-routing if quality drops.
