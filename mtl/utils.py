import json
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score


def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def flatten_and_concat(gradients):
    flat = []
    for g in gradients:
        if g is None:
            continue
        flat.append(tf.reshape(g, [-1]))
    if not flat:
        return tf.zeros([0], dtype=tf.float32)
    return tf.concat(flat, axis=0)


def fill_none_gradients(gradients, variables):
    return [g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, variables)]


def cosine_similarity(gradients_a, gradients_b, eps=1e-8):
    a = flatten_and_concat(gradients_a)
    b = flatten_and_concat(gradients_b)
    dot = tf.reduce_sum(a * b)
    denom = tf.norm(a) * tf.norm(b)
    return dot / (denom + eps)


def pcgrad(gradients_a, gradients_b, eps=1e-8):
    a = []
    b = []
    for ga, gb in zip(gradients_a, gradients_b):
        if ga is None and gb is None:
            a.append(tf.zeros([1], dtype=tf.float32))
            b.append(tf.zeros([1], dtype=tf.float32))
            continue
        if ga is None:
            ga = tf.zeros_like(gb)
        if gb is None:
            gb = tf.zeros_like(ga)
        a.append(ga)
        b.append(gb)

    proj_a = []
    proj_b = []
    for ga, gb in zip(a, b):
        dot_ab = tf.reduce_sum(ga * gb)
        gb_norm2 = tf.reduce_sum(gb * gb)
        ga_norm2 = tf.reduce_sum(ga * ga)
        proj_a.append(ga - (dot_ab / (gb_norm2 + eps)) * gb)
        proj_b.append(gb - (dot_ab / (ga_norm2 + eps)) * ga)
    return proj_a, proj_b


def write_metrics_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def evaluate_model(model, dataset):
    y_true_a = []
    y_true_b = []
    y_pred_a = []
    y_pred_b = []

    for x_batch, (y_a_batch, y_b_batch) in dataset:
        pred_a, pred_b = model(x_batch, training=False)
        y_true_a.extend(y_a_batch.numpy().tolist())
        y_true_b.extend(y_b_batch.numpy().tolist())
        y_pred_a.extend((pred_a.numpy().reshape(-1) >= 0.5).astype(int).tolist())
        y_pred_b.extend((pred_b.numpy().reshape(-1) >= 0.5).astype(int).tolist())

    metrics = {
        "accuracy_a": accuracy_score(y_true_a, y_pred_a),
        "f1_a": f1_score(y_true_a, y_pred_a),
        "accuracy_b": accuracy_score(y_true_b, y_pred_b),
        "f1_b": f1_score(y_true_b, y_pred_b),
    }
    return metrics


def update_final_metrics(path, experiment_name, metrics):
    path = pathlib.Path(path)
    final = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            final = json.load(f)
    final[experiment_name] = metrics
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
