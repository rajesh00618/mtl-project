import tensorflow as tf

IMG_SHAPE = (28, 28, 1)


def build_task_labels(y):
    y = tf.cast(y, tf.int32)
    label_a = tf.cast(y % 2, tf.int32)
    label_b = tf.cast(y > 4, tf.int32)
    return label_a, label_b


def load_mnist_dataset(batch_size=64, buffer_size=10000):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    train_a, train_b = build_task_labels(y_train)
    test_a, test_b = build_task_labels(y_test)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, (train_a, train_b)))
    train_ds = train_ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, (test_a, test_b)))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def get_visualization_sample(sample_size=2048):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x = x_train.astype("float32") / 255.0
    x = x[..., None]
    label_a, label_b = build_task_labels(y_train)
    return x[:sample_size], label_a[:sample_size], label_b[:sample_size]
