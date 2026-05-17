import csv
import pathlib

import tensorflow as tf

from mtl.data import load_mnist_dataset
from mtl.model import MultiTaskModel
from mtl.utils import cosine_similarity, ensure_dir, evaluate_model, fill_none_gradients, pcgrad, update_final_metrics


RESULTS_DIR = pathlib.Path("results")
ensure_dir(RESULTS_DIR)


def build_dataset(batch_size=128):
    return load_mnist_dataset(batch_size=batch_size)


@tf.function
def train_step(model, optimizer, loss_fn, x_batch, y_a_batch, y_b_batch):
    with tf.GradientTape(persistent=True) as tape:
        pred_a, pred_b = model(x_batch, training=True)
        loss_a = loss_fn(y_a_batch, pred_a)
        loss_b = loss_fn(y_b_batch, pred_b)

    backbone_vars = model.backbone.trainable_variables
    head_a_vars = model.head_a.trainable_variables
    head_b_vars = model.head_b.trainable_variables

    grads_a_backbone = tape.gradient(loss_a, backbone_vars)
    grads_b_backbone = tape.gradient(loss_b, backbone_vars)
    grads_a_head = tape.gradient(loss_a, head_a_vars)
    grads_b_head = tape.gradient(loss_b, head_b_vars)
    del tape

    cos_sim = cosine_similarity(grads_a_backbone, grads_b_backbone)
    proj_a, proj_b = pcgrad(grads_a_backbone, grads_b_backbone)
    final_backbone = [ga + gb for ga, gb in zip(proj_a, proj_b)]

    grads_a_head = fill_none_gradients(grads_a_head, head_a_vars)
    grads_b_head = fill_none_gradients(grads_b_head, head_b_vars)
    final_grads = final_backbone + grads_a_head + grads_b_head
    final_vars = backbone_vars + head_a_vars + head_b_vars
    optimizer.apply_gradients(zip(final_grads, final_vars))

    return loss_a, loss_b, cos_sim


def main():
    train_ds, val_ds = build_dataset()
    model = MultiTaskModel()
    _ = model(tf.zeros((1, 28, 28, 1), dtype=tf.float32))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    history = []
    gradient_rows = []
    step = 0
    epochs = 5
    for epoch in range(1, epochs + 1):
        for x_batch, (y_a_batch, y_b_batch) in train_ds:
            loss_a, loss_b, cos_sim = train_step(model, optimizer, loss_fn, x_batch, y_a_batch, y_b_batch)
            step += 1
            gradient_rows.append({"step": step, "cosine_similarity": float(cos_sim.numpy())})

        val_metrics = evaluate_model(model, val_ds)
        row = {
            "epoch": epoch,
            "val_accuracy_a": val_metrics["accuracy_a"],
            "val_f1_a": val_metrics["f1_a"],
            "val_accuracy_b": val_metrics["accuracy_b"],
            "val_f1_b": val_metrics["f1_b"],
        }
        history.append(row)
        print(f"PCGrad epoch {epoch}: {row}")

    gradient_path = RESULTS_DIR / "gradient_conflict.csv"
    with open(gradient_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["step", "cosine_similarity"])
        writer.writeheader()
        writer.writerows(gradient_rows)

    pcgrad_path = RESULTS_DIR / "pcgrad_metrics.csv"
    with open(pcgrad_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    model.save_weights(RESULTS_DIR / "pcgrad_model.weights")
    update_final_metrics(RESULTS_DIR / "final_metrics.json", "pcgrad", {
        "task_a": {
            "accuracy": history[-1]["val_accuracy_a"],
            "f1_score": history[-1]["val_f1_a"],
        },
        "task_b": {
            "accuracy": history[-1]["val_accuracy_b"],
            "f1_score": history[-1]["val_f1_b"],
        },
    })
    print(f"Saved PCGrad metrics to {pcgrad_path}")


if __name__ == "__main__":
    main()
