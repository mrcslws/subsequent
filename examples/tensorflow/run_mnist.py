from functools import partial

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

from subsequent.tensorflow.conv2d_model import RecurrentDynamicCNN


def baseline_static_experiment():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    print(model.summary())

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=15, batch_size=512)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_acc)


def normalize(x, mean, stdev):
    return (x - mean) / stdev

MEAN = 0.13062755
STDEV = 0.30810780


def adaptive_experiment():
    model = models.Sequential(
        layers=[
            RecurrentDynamicCNN(
                channels=(6,6,6),
                num_state_units=64,
                static_weight_decay=3e-4,
                dynamic_weight_decay=1e-3,
                num_iterations=6,
                num_mlp_units=64,
                num_mlp_hidden_layers=1,
                num_embedding_to_params_hidden_layers=0,
                out_channel_embedding_sizes=(10, 32, 32, 32),
                initialize_embeddings=True,
                embedding_initializer="glorot_uniform",
                layernorm_residual_stream=True,
                use_batchnorm=False,
                reuse_norm=True,
                combine_mode="gru",
                integrate_initial_state=True,
                gru_include_query_mode="all",
                single_gru_mode=False,
                classifier_num_iterations=2,
            ),
            layers.Dense(10, name="classifier"),
        ],
    )

    model.build(input_shape=(None, 28, 28, 1))
    print(model.summary(expand_nested=True))

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    train_images = normalize(train_images, MEAN, STDEV)
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255
    test_images = normalize(test_images, MEAN, STDEV)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-8),
                  loss=partial(tf.keras.metrics.sparse_categorical_crossentropy, from_logits=True),
                  metrics=['accuracy'])

    step_size = 44
    alpha = 0.3

    def scheduler(epoch, lr):
        prev_stage = (epoch - 1) // step_size
        stage = epoch // step_size
        if stage > prev_stage and stage > 0:
            return lr * alpha
        return lr

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.fit(train_images, train_labels, epochs=45, batch_size=512,
              callbacks=[callback],
              validation_freq=1,
              validation_data=(test_images, test_labels)
              )


if __name__ == "__main__":
    adaptive_experiment()
