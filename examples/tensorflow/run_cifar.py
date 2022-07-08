import argparse
import os
import random
import time
from datetime import datetime
from filelock import FileLock
from functools import partial
from threading import Thread
from typing import Dict, List, Union

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist, cifar10

from subsequent.tensorflow.conv2d_model import CIFAR10_CNN4_CONFIG, RecurrentDynamicCNN


def adaptive_experiment(config):

    print(config)

    config2 = dict(
        learning_rate=0.001,
        batch_size=512,
        epochs=45,
        lr_schedule_step_size=44,
        lr_schedule_alpha=0.3,
        conv_channels=(6,6,6),
        state_size=64,
        mlp_weight_decay=3e-4,
        dynamic_conv_weight_decay=1e-3,
        num_iterations=2,
        num_mlp_units=64,
        num_mlp_hidden_layers=2,
        num_embedding_to_params_hidden_layers=0,
        out_channel_embedding_sizes=(20, 32, 32, 32),
        initialize_embeddings=True,
        embedding_initializer="glorot_uniform",
        layernorm_residual_stream=True,
        use_batchnorm=False,
        reuse_norm=True,
        recurrent=True,
        combine_mode="gru",
        integrate_initial_state=True,
        gru_include_query_mode="all",
        single_gru_mode=False,
        classifier_num_iterations=2,
    )

    config2.update(config)
    config = config2
    del config2

    model_class = RecurrentDynamicCNN

    model = models.Sequential(
        layers=[
            model_class(
                num_iterations=config["num_iterations"],
                num_state_units=config["state_size"],
                num_mlp_units=config["num_mlp_units"],
                num_mlp_hidden_layers=config["num_mlp_hidden_layers"],
                out_channel_embedding_sizes=config["out_channel_embedding_sizes"],
                channels=config["conv_channels"],
                static_weight_decay=config["mlp_weight_decay"],
                dynamic_weight_decay=config["dynamic_conv_weight_decay"],
                num_embedding_to_params_hidden_layers=config["num_embedding_to_params_hidden_layers"],
                initialize_embeddings=config["initialize_embeddings"],
                embedding_initializer=config["embedding_initializer"],
                layernorm_residual_stream=config["layernorm_residual_stream"],
                use_batchnorm=config["use_batchnorm"],
                reuse_norm=config["reuse_norm"],
                combine_mode=config["combine_mode"],
                integrate_initial_state=config["integrate_initial_state"],
                gru_include_query_mode=config["gru_include_query_mode"],
                single_gru_mode=config["single_gru_mode"],
                classifier_num_iterations=config["classifier_num_iterations"],
                network_config=CIFAR10_CNN4_CONFIG,
                num_data_channels=3,
            ),
            layers.Dense(10, name="classifier"),
        ],
    )

    model.build(input_shape=(None, 32, 32, 3))
    print(model.summary(expand_nested=True))

    with FileLock(os.path.expanduser("~/.data.lock")):
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.reshape((50000, 32, 32, 3))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 32, 32, 3))
    test_images = test_images.astype('float32') / 255
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"],
                                                     epsilon=1e-8),
                  loss=partial(tf.keras.metrics.sparse_categorical_crossentropy,
                               from_logits=True),
                  steps_per_execution=20,
                  metrics=['accuracy']
                  )

    step_size = config["lr_schedule_step_size"]
    alpha = config["lr_schedule_alpha"]

    def scheduler(epoch, lr):
        prev_stage = (epoch - 1) // step_size
        stage = epoch // step_size
        if stage > prev_stage and stage > 0:
            return lr * alpha
        return lr

    # logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
    #                                                  # profile_batch = "10,20"
    #                                                  )


    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.fit(train_images, train_labels,
              epochs=config["epochs"],
              batch_size=config["batch_size"],
              callbacks=[callback,
                         # tboard_callback
                         ],
              validation_data=(test_images, test_labels),
              )


if __name__ == "__main__":
    adaptive_experiment({})
