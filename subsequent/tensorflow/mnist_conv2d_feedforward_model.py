"""
See conv2d_model.py for the latest code, but it does not yet support
feedforward mode. (It instead uses the same weights on every iteration.)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers


STATE_SIZE = 16
STATIC_WD = 1e-3
DYNAMIC_WD = 1e-3

CONV_CHANNELS = (3, 6, 3)
NUM_MLP_UNITS = 16


def channelize_batch_of_weights(weight, h, w, num_in_channels, num_out_channels):
    weight = tf.reshape(weight, [-1, num_out_channels, h * w * num_in_channels])
    weight = tf.transpose(weight, [2, 0, 1])
    return tf.reshape(weight, [h, w, num_in_channels, -1])


class DynamicCNN4(tf.keras.Model):
    def __init__(self, num_mlp_units=NUM_MLP_UNITS, channels=CONV_CHANNELS,
                 num_state_units=STATE_SIZE, static_weight_decay=STATIC_WD,
                 dynamic_weight_decay=DYNAMIC_WD, mlp_dropout_rate=0.):
        super().__init__()

        self.num_input_channels = 1
        (self.num_channels1,
         self.num_channels2,
         self.num_channels3) = channels
        self.num_state_units = num_state_units
        self.splits = (9 * self.num_channels1 * 1,
                       self.num_channels1,
                       9 * self.num_channels2 * self.num_channels1,
                       self.num_channels2,
                       9 * self.num_channels3 * self.num_channels2,
                       self.num_channels3,
                       9 * num_state_units * self.num_channels3,
                       num_state_units)
        self.mlp = tf.keras.Sequential(
            layers=[layers.Dense(num_mlp_units, activation="relu",
                                 input_shape=(num_state_units,),
                                 kernel_regularizer=regularizers.L2(static_weight_decay),
                                 ),
                    layers.Dropout(mlp_dropout_rate),
                    layers.Dense(num_mlp_units, activation="relu",
                                 kernel_regularizer=regularizers.L2(static_weight_decay),
                                 ),
                    layers.Dropout(mlp_dropout_rate),
                    layers.Dense(sum(self.splits),
                                 kernel_regularizer=regularizers.L2(static_weight_decay),
                                 activity_regularizer=regularizers.L2(dynamic_weight_decay))]
        )

    def call(self, image, state):
        w = self.mlp(state)
        (weight1,
         bias1,
         weight2,
         bias2,
         weight3,
         bias3,
         weight4,
         bias4) = tf.split(w, self.splits, axis=1)

        weight1 = channelize_batch_of_weights(
            weight1, 3, 3, self.num_input_channels, self.num_channels1)
        bias1 = tf.reshape(bias1, [-1])
        x = tf.nn.conv2d(image, weight1, strides=1, padding="VALID") + bias1
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, 2, 2, "VALID")

        weight2 = channelize_batch_of_weights(
            weight2, 3, 3, self.num_channels1, self.num_channels2)
        bias2 = tf.reshape(bias2, [-1])
        x = tf.nn.conv2d(x, weight2, strides=1, padding="VALID") + bias2
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, 2, 2, "VALID")

        weight3 = channelize_batch_of_weights(
            weight3, 3, 3, self.num_channels2, self.num_channels3)
        bias3 = tf.reshape(bias3, [-1])
        x = tf.nn.conv2d(x, weight3, strides=1, padding="VALID") + bias3
        x = tf.nn.relu(x)

        weight4 = channelize_batch_of_weights(
            weight4, 3, 3, self.num_channels3, self.num_state_units)
        bias4 = tf.reshape(bias4, [-1])
        x = tf.nn.conv2d(x, weight4, strides=1, padding="VALID") + bias4

        x = tf.transpose(x, [3, 1, 2, 0])
        return state + tf.reshape(x, [-1, self.num_state_units])


def stem4(input_shape=(28, 28, 1), channels=CONV_CHANNELS, num_state_units=STATE_SIZE,
          weight_decay=STATIC_WD):
    return tf.keras.Sequential(
        layers=[layers.Conv2D(channels[0], 3, activation="relu",
                              input_shape=input_shape,
                              kernel_regularizer=regularizers.L2(weight_decay),
                              ),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(channels[1], 3, activation="relu",
                              kernel_regularizer=regularizers.L2(weight_decay),
                              ),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(channels[2], 3, activation="relu",
                              kernel_regularizer=regularizers.L2(weight_decay),
                              ),
                layers.Flatten(),
                layers.Dense(num_state_units, activation='relu',
                             kernel_regularizer=regularizers.L2(weight_decay))]
    )


class IterativeDynamicCNN(tf.keras.Model):
    def __init__(self, num_iterations=2, num_state_units=STATE_SIZE,
                 num_mlp_hidden_layers=2,
                 num_mlp_units=NUM_MLP_UNITS, channels=CONV_CHANNELS,
                 out_channel_embedding_sizes=(6, 8, 8, 8),
                 static_weight_decay=STATIC_WD, dynamic_weight_decay=DYNAMIC_WD,
                 num_embedding_to_params_hidden_units=0,
                 num_embedding_to_params_hidden_layers=0,
                 layernorm_residual_stream=True,
                 initialize_embeddings=False,
                 embedding_initializer="glorot_uniform",
                 use_batchnorm=False,
                 reuse_norm=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_iterations = num_iterations
        self.num_state_units = num_state_units
        self.layernorm_residual_stream = layernorm_residual_stream

        self.reuse_norm = reuse_norm
        if reuse_norm:
            self.ln = (layers.BatchNormalization()
                       if use_batchnorm
                       else layers.LayerNormalization(epsilon=1e-5))
        else:
            self.lns = [(layers.BatchNormalization()
                         if use_batchnorm
                         else layers.LayerNormalization(epsilon=1e-5))
                        for _ in range(num_iterations + 1)]

        self.num_in_channels = (1,) + channels
        self.num_out_channels = channels + (num_state_units,)
        self.out_channel_embedding_sizes = out_channel_embedding_sizes
        self.num_weights_per_out_channel = [
            3 * 3 * self.num_in_channels[i]
            for i in range(4)
        ]

        self.embedding_splits = (
            self.num_out_channels[0] * out_channel_embedding_sizes[0],
            self.num_out_channels[1] * out_channel_embedding_sizes[1],
            self.num_out_channels[2] * out_channel_embedding_sizes[2],
            self.num_out_channels[3] * out_channel_embedding_sizes[3])
        self.mlps = [
            tf.keras.Sequential(
                layers=[
                    layers.Dense(
                        num_mlp_units, activation="relu",
                        input_shape=(num_state_units,),
                        kernel_regularizer=regularizers.L2(static_weight_decay))
                    for _ in range(num_mlp_hidden_layers)]
                + [layers.Dense(
                    sum(self.embedding_splits),
                    kernel_regularizer=regularizers.L2(static_weight_decay),
                )],
                name=f"state_to_out_channel_embeddings{it}"
            )
            for it in range(self.num_iterations)
        ]

        self.embedding_to_params = [
            tf.keras.Sequential(
                layers=[
                    layers.Dense(
                        num_embedding_to_params_hidden_units,
                        activation="relu",
                        kernel_regularizer=regularizers.L2(static_weight_decay)
                    )
                    for _ in range(num_embedding_to_params_hidden_layers)]
                        + [
                    layers.Dense(
                        self.num_in_channels[i] * 3 * 3 + 1,
                        kernel_regularizer=regularizers.L2(static_weight_decay),
                        activity_regularizer=regularizers.L2(dynamic_weight_decay),
                        # Either these biases or the stem_embeddings need to be
                        # initialized to nonzero. Otherwise, relu will mask the
                        # gradients and prevent the trained stem_embeddings from ever
                        # being updated.
                        bias_initializer=("zeros" if initialize_embeddings else embedding_initializer))
                    ],
                name=f"out_channel_embedding_to_params{i}")
            for i in range(4)]

        # Stem CNN compressed into an embedding
        self.stem_embeddings = [tf.Variable((tf.zeros([self.num_out_channels[i],
                                                       out_channel_embedding_sizes[i]])
                                             if not initialize_embeddings
                                             else (tf.keras.initializers.GlorotUniform
                                                   if embedding_initializer == "glorot_uniform"
                                                   else tf.keras.initializers.GlorotNormal)()(shape=(self.num_out_channels[i],
                                                                                                     out_channel_embedding_sizes[i]))),
                                            trainable=True)
                                for i in range(4)]


    def call(self, image):
        # First layer: apply same conv layer to every item in batch
        x = image
        for i in range(4):
            # self.stem_embeddings[i] is an embedding vector for every out channel
            conv_params = self.embedding_to_params[i](self.stem_embeddings[i])

            # num_output_channels[i] param vectors, now split into weights and biases
            w, b = tf.split(conv_params, [self.num_weights_per_out_channel[i], 1],
                            axis=1)
            # move out_channels to last dimension
            w = tf.transpose(w)
            w = tf.reshape(w, (3, 3, self.num_in_channels[i], self.num_out_channels[i]))
            b = tf.reshape(b, [-1])
            x = tf.nn.conv2d(x, w, strides=1, padding="VALID") + b

            if i < 3:
                x = tf.nn.relu(x)

            if i < 2:
                x = tf.nn.max_pool2d(x, 2, 2, "VALID")

        state = tf.reshape(x, (-1, self.num_state_units))
        del x
        ln = (self.ln if self.reuse_norm else self.lns[0])
        state = ln(state)

        # Subsequent layers: Apply different conv layers to each item in batch.
        # Do a group convolution, arranging batch items as different channels of
        # a single image.
        image = tf.transpose(image, [3, 1, 2, 0])

        for layer_i, embedding_mlp in enumerate(self.mlps):
            # batch of full embedding vectors
            conv_embeddings = embedding_mlp(state)

            # (conv1_embeddings, conv2_embeddings, ..., conv4_embeddings) where
            # each is a batch
            conv_embeddings = tf.split(conv_embeddings, self.embedding_splits, axis=1)

            x = image
            for i in range(4):
                # layer_conv_embeddings: batch of embeddings for this layer

                # Contains batch_size * num_output_channels[i] embeddings
                layer_conv_embeddings = tf.reshape(conv_embeddings[i],
                                                   (-1, self.out_channel_embedding_sizes[i]))

                # Contains batch_size * num_output_channels[i] param vectors
                conv_params = self.embedding_to_params[i](layer_conv_embeddings)

                # batch_size * num_output_channels[i] param vectors, now split into weights and biases
                w, b = tf.split(conv_params, [self.num_weights_per_out_channel[i], 1],
                                axis=1)
                # w is batch_size flattened complete weight tensors
                # b doesn't need similar treatment, since we're going to treat batch items as channels
                w = tf.reshape(w, (-1, 3 * 3 * self.num_out_channels[i] * self.num_in_channels[i]))
                # w is a weight tensor with (batch_size * num_output_channels[i])
                # out channels and num_input_channels[i] in channels, so it
                # performs a group convolution with batch_size groups
                w = channelize_batch_of_weights(w, 3, 3, self.num_in_channels[i],
                                                self.num_out_channels[i])
                b = tf.reshape(b, [-1])

                x = tf.nn.conv2d(x, w, strides=1, padding="VALID") + b

                if i < 3:
                    x = tf.nn.relu(x)

                if i < 2:
                    x = tf.nn.max_pool2d(x, 2, 2, "VALID")

            x = tf.reshape(x, [-1, self.num_state_units])
            ln = (self.ln if self.reuse_norm else self.lns[layer_i + 1])
            state = (ln(state + x)
                     if self.layernorm_residual_stream
                     else state + ln(x))
            del x

        return state
