"""Train and evaluate a reproducible FashionFinder classification experiment."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data import CLASS_NAMES, load_fashion_mnist, make_dataset
from src.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "cnn"), default="baseline")
    parser.add_argument("--run-name", required=True, help="Folder name under artifacts/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Optional cap for a quick local smoke test; do not use for final metrics.",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="Optional cap for a quick local smoke test; do not use for final metrics.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.80,
        help="Send predictions below this confidence to human review.",
    )
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow as far as the runtime permits."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def limit_split(
    split: tuple[np.ndarray, np.ndarray], limit: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally cut a split for a fast workflow check."""
    images, labels = split
    if limit is None:
        return images, labels
    if limit <= 0:
        raise ValueError("Dataset limits must be positive")
    return images[:limit], labels[:limit]


def class_recall(confusion_matrix: np.ndarray) -> dict[str, float]:
    """Return per-category recall from a row=true, column=predicted matrix."""
    row_totals = confusion_matrix.sum(axis=1)
    return {
        CLASS_NAMES[index]: float(confusion_matrix[index, index] / row_totals[index])
        if row_totals[index]
        else 0.0
        for index in range(len(CLASS_NAMES))
    }

def routing_metrics(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float | None]:
    """Calculate review and auto-routing rates for a confidence threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence-threshold must be between 0 and 1")

    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    auto_route = confidence >= threshold

    auto_route_accuracy = (
        float(np.mean(predictions[auto_route] == labels[auto_route]))
        if np.any(auto_route)
        else None
    )

    return {
        "review_rate": float(np.mean(~auto_route)),
        "auto_route_rate": float(np.mean(auto_route)),
        "auto_route_accuracy": auto_route_accuracy,
    }
    

def save_confusion_matrix(confusion_matrix: np.ndarray, destination: Path) -> None:
    """Save a human-readable confusion matrix with category labels."""
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["actual\\predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, confusion_matrix, strict=True):
            writer.writerow([class_name, *row.tolist()])


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence-threshold must be between 0 and 1")
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")

    run_dir = Path(args.artifacts_dir) / args.run_name
    if run_dir.exists():
        raise FileExistsError(
            f"{run_dir} already exists. Choose a new --run-name to preserve experiment evidence."
        )
    run_dir.mkdir(parents=True)
    set_reproducible_seed(args.seed)

    train_split, validation_split, test_split = load_fashion_mnist(
        data_dir=args.data_dir, validation_size=args.validation_size, seed=args.seed
    )
    train_split = limit_split(train_split, args.limit_train)
    test_split = limit_split(test_split, args.limit_test)
    x_train, y_train = train_split
    x_validation, y_validation = validation_split
    x_test, y_test = test_split

    model = build_model(args.model, learning_rate=args.learning_rate)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"), monitor="val_accuracy", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_history.csv")),
    ]
    history = model.fit(
        make_dataset(x_train, y_train, args.batch_size, training=True, seed=args.seed),
        validation_data=make_dataset(x_validation, y_validation, args.batch_size),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(
        make_dataset(x_test, y_test, args.batch_size), verbose=0
    )
    probabilities = model.predict(make_dataset(x_test, y_test, args.batch_size), verbose=0)
    
    routing = routing_metrics(probabilities, y_test, args.confidence_threshold)
    
    predictions = np.argmax(probabilities, axis=1)
    confusion = tf.math.confusion_matrix(y_test, predictions, num_classes=len(CLASS_NAMES)).numpy()
    recalls = class_recall(confusion)
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1

    metrics = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "parameters": int(model.count_params()),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_examples": len(x_train),
        "validation_examples": len(x_validation),
        "test_examples": len(x_test),
        "best_validation_accuracy": float(max(history.history["val_accuracy"])),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "class_recall": recalls,
        "worst_class_recall": float(min(recalls.values())),
        "routing": routing,
        "confidence_threshold": args.confidence_threshold,
        **routing,
        "note": "Add confidence-threshold routing metrics as part of the engineering task.",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(confusion, run_dir / "confusion_matrix.csv")
    model.save(run_dir / "final_model.keras")

    print(f"Run saved to: {run_dir}")
    print(f"Best validation accuracy: {metrics['best_validation_accuracy']:.2%}")
    print(f"Test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"Worst-category recall: {metrics['worst_class_recall']:.2%}")


if __name__ == "__main__":
    
    main()
