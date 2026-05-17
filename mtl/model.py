import tensorflow as tf


class MultiTaskModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.backbone = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.3),
            ],
            name="backbone",
        )

        self.head_a = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
            name="head_a",
        )

        self.head_b = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
            name="head_b",
        )

    def call(self, inputs, training=False):
        shared = self.backbone(inputs, training=training)
        return self.head_a(shared, training=training), self.head_b(shared, training=training)
