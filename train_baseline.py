import csv
import pathlib

import tensorflow as tf

from mtl.data import load_mnist_dataset
from mtl.model import MultiTaskModel
from mtl.utils import ensure_dir, evaluate_model, update_final_metrics


RESULTS_DIR = pathlib.Path("results")
ensure_dir(RESULTS_DIR)


def build_dataset(batch_size=128):
    return load_mnist_dataset(batch_size=batch_size)


@tf.function
def train_step(model, optimizer, loss_fn, x_batch, y_a_batch, y_b_batch):
    with tf.GradientTape() as tape:
        pred_a, pred_b = model(x_batch, training=True)
        loss_a = loss_fn(y_a_batch, pred_a)
        loss_b = loss_fn(y_b_batch, pred_b)
        loss = loss_a + loss_b
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_a, loss_b, loss


def main():
    train_ds, val_ds = build_dataset()
    model = MultiTaskModel()
    _ = model(tf.zeros((1, 28, 28, 1), dtype=tf.float32))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    history = []
    epochs = 5
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        steps = 0
        for x_batch, (y_a_batch, y_b_batch) in train_ds:
            loss_a, loss_b, loss = train_step(model, optimizer, loss_fn, x_batch, y_a_batch, y_b_batch)
            epoch_loss += loss.numpy()
            steps += 1

        val_metrics = evaluate_model(model, val_ds)
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, steps),
            "val_accuracy_a": val_metrics["accuracy_a"],
            "val_f1_a": val_metrics["f1_a"],
            "val_accuracy_b": val_metrics["accuracy_b"],
            "val_f1_b": val_metrics["f1_b"],
        }
        history.append(row)
        print(f"Baseline epoch {epoch}: {row}")

    baseline_path = RESULTS_DIR / "baseline_metrics.csv"
    with open(baseline_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    model.save_weights(RESULTS_DIR / "baseline_model.weights")
    update_final_metrics(RESULTS_DIR / "final_metrics.json", "baseline", {
        "task_a": {
            "accuracy": history[-1]["val_accuracy_a"],
            "f1_score": history[-1]["val_f1_a"],
        },
        "task_b": {
            "accuracy": history[-1]["val_accuracy_b"],
            "f1_score": history[-1]["val_f1_b"],
        },
    })
    print(f"Saved baseline metrics to {baseline_path}")


if __name__ == "__main__":
    main()
