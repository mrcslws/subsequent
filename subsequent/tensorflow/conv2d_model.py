from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers



STATE_SIZE = 16
STATIC_WD = 1e-3
DYNAMIC_WD = 1e-3

CONV_CHANNELS = (3, 6, 3)
NUM_MLP_UNITS = 16


def channelize_batch_of_weights(weight, fh, fw, num_in_channels, num_out_channels):
    weight = tf.reshape(weight, [-1, num_out_channels, fh * fw * num_in_channels])
    weight = tf.transpose(weight, [2, 0, 1])
    return tf.reshape(weight, [fh, fw, num_in_channels, -1])




MNIST_CNN4_CONFIG = [
    (3, 3, 1, "VALID",
     [tf.nn.relu,
      partial(tf.nn.max_pool2d, ksize=2, strides=2, padding="VALID")]),
    (3, 3, 1, "VALID",
     [tf.nn.relu,
      partial(tf.nn.max_pool2d, ksize=2, strides=2, padding="VALID")]),
    (3, 3, 1, "VALID",
     [tf.nn.relu]),
    (3, 3, 1, "VALID",
     [])]

CIFAR10_CNN4_CONFIG = [
    (3, 3, 1, "VALID",
     [tf.nn.relu,
      partial(tf.nn.max_pool2d, ksize=2, strides=2, padding="VALID")]),
    (3, 3, 1, "VALID",
     [tf.nn.relu,
      partial(tf.nn.max_pool2d, ksize=2, strides=2, padding="VALID")]),
    (3, 3, 1, "VALID",
     [tf.nn.relu]),
    (4, 4, 1, "VALID",
     [])]



class IterativeLayer(tf.keras.Model):
    def __init__(self, num_iterations=2,
                 num_state_units=STATE_SIZE,
                 num_mlp_hidden_layers=2,
                 num_mlp_units=NUM_MLP_UNITS,
                 channels=CONV_CHANNELS,
                 out_channel_embedding_sizes=(6, 8, 8, 8),
                 static_weight_decay=STATIC_WD, dynamic_weight_decay=DYNAMIC_WD,
                 num_embedding_to_params_hidden_layers=0,
                 layernorm_residual_stream=True,
                 initialize_embeddings=False,
                 embedding_initializer="glorot_uniform",
                 use_batchnorm=False,
                 reuse_norm=True,
                 combine_mode="residual",
                 integrate_initial_state=True,
                 gru_include_query_mode="attentional",
                 network_config=MNIST_CNN4_CONFIG,
                 num_data_channels=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_iterations = num_iterations
        self.num_state_units = num_state_units
        self.layernorm_residual_stream = layernorm_residual_stream
        self.integrate_initial_state = integrate_initial_state
        self.gru_include_query_mode = gru_include_query_mode
        assert reuse_norm


        self.network_config = network_config

        self.ln = (layers.BatchNormalization()
                   if use_batchnorm
                   else layers.LayerNormalization(epsilon=1e-5))

        self.num_in_channels = (num_data_channels,) + channels
        self.num_out_channels = channels + (num_state_units,)
        self.out_channel_embedding_sizes = out_channel_embedding_sizes

        self.embedding_splits = [
            self.num_out_channels[i] * out_channel_embedding_sizes[i]
            for i, _ in enumerate(self.network_config)]

        self.num_mlp_hidden_layers = num_mlp_hidden_layers
        if num_mlp_hidden_layers > 0:
            self.state_to_query = tf.keras.Sequential(
                layers=[
                    layers.Dense(
                        num_mlp_units, activation="relu",
                        input_shape=(num_state_units,),
                        kernel_regularizer=regularizers.L2(static_weight_decay))
                    for _ in range(num_mlp_hidden_layers)],
                name="state_to_query"
            )
        else:
            # Identity
            self.state_to_query = layers.Layer(name="state_to_query",
                                               input_shape=(num_state_units,)
                                               )

        self.query_to_embeddings = layers.Dense(
            sum(self.embedding_splits),
            kernel_regularizer=regularizers.L2(static_weight_decay),
            name="query_to_embeddings"
        )

        self.embedding_to_params = [
            tf.keras.Sequential(
                layers=[
                    layers.Dense(
                        out_channel_embedding_sizes[i],
                        activation="relu",
                        kernel_regularizer=regularizers.L2(static_weight_decay)
                    )
                    for _ in range(num_embedding_to_params_hidden_layers)]
                        + [
                    layers.Dense(
                        self.num_in_channels[i] * fh * fw + 1,
                        kernel_regularizer=regularizers.L2(static_weight_decay),
                        activity_regularizer=regularizers.L2(dynamic_weight_decay),
                        # Either these biases or the stem_embeddings need to be
                        # initialized to nonzero. Otherwise, relu will mask the
                        # gradients and prevent the trained stem_embeddings from ever
                        # being updated.
                        bias_initializer=("zeros" if initialize_embeddings else embedding_initializer))
                    ],
                name=f"out_channel_embedding_to_params{i}")
            for i, (fh, fw, *ignored_args) in enumerate(self.network_config)]

        self.combine_mode = combine_mode
        if combine_mode == "gru" and (integrate_initial_state or num_iterations > 0):
            self.sigmoid_layer_z = layers.Dense(
                self.num_state_units,
                activation="sigmoid",
                name="sigmoid_z",
                kernel_regularizer=regularizers.L2(static_weight_decay))
            self.sigmoid_layer_r = layers.Dense(
                self.num_state_units,
                activation="sigmoid",
                name="sigmoid_r",
                kernel_regularizer=regularizers.L2(static_weight_decay))
            self.linear_value_layer = layers.Dense(
                self.num_state_units,
                name="gru_value",
                kernel_regularizer=regularizers.L2(static_weight_decay))

    def call(self, image, state):
        """
        Apply different conv layers to each item in batch.
        Do a group convolution, arranging batch items as different channels of
        a single image.
        """
        for iteration in range(self.num_iterations):
            # batch of query vectors
            query = self.state_to_query(state)

            # batch of full embedding vectors
            conv_embeddings = self.query_to_embeddings(query)

            # (conv1_embeddings, conv2_embeddings, ..., conv4_embeddings) where
            # each is a batch
            conv_embeddings = tf.split(conv_embeddings, self.embedding_splits, axis=1)

            x = image

            for i, config in enumerate(self.network_config):
                fh, fw, strides, padding, fs = config
                num_out_channels = self.num_out_channels[i]
                num_in_channels = self.num_in_channels[i]

                # layer_conv_embeddings: batch of embeddings for this layer

                # Contains batch_size * num_output_channels[i] embeddings
                layer_conv_embeddings = tf.reshape(conv_embeddings[i],
                                                   (-1, self.out_channel_embedding_sizes[i]))

                # Contains batch_size * num_output_channels[i] param vectors
                conv_params = self.embedding_to_params[i](layer_conv_embeddings)

                # batch_size * num_output_channels[i] param vectors, now split into weights and biases
                num_weights_per_out_channel = fh * fw * num_in_channels
                w, b = tf.split(conv_params, [num_weights_per_out_channel, 1],
                                axis=1)
                # w is batch_size flattened complete weight tensors
                # b doesn't need similar treatment, since we're going to treat batch items as channels
                w = tf.reshape(w, (-1, fh * fw * num_out_channels * num_in_channels))
                # w is a weight tensor with (batch_size * num_output_channels[i])
                # out channels and num_input_channels[i] in channels, so it
                # performs a group convolution with batch_size groups
                w = channelize_batch_of_weights(w, fh, fw, num_in_channels, num_out_channels)
                b = tf.reshape(b, [-1])

                x = tf.nn.conv2d(x, w, strides=strides, padding=padding) + b

                # TODO consider performing layernorm

                for f in fs:
                    x = f(x)

            x = tf.reshape(x, [-1, self.num_state_units])

            if self.combine_mode == "residual":
                state = (self.ln(state + x)
                         if self.layernorm_residual_stream
                         else state + self.ln(x))
            elif self.combine_mode == "gru":
                prev_concat = tf.concat([state, x]
                                        + ([query]
                                           if self.gru_include_query_mode in ("attentional", "all")
                                           and self.num_mlp_hidden_layers > 0
                                           else []),
                                        1)
                z = self.sigmoid_layer_z(prev_concat)
                r = self.sigmoid_layer_r(prev_concat)
                h_tilde = self.linear_value_layer(tf.concat([r * state, x] +
                                                            ([query]
                                                             if self.gru_include_query_mode == "all"
                                                             and self.num_mlp_hidden_layers > 0
                                                             else []),
                                                            1))
                state = (1 - z) * state + z * h_tilde
                state = self.ln(state)
            else:
                raise ValueError(self.combine_mode)
            del x

        return state

    def call_stem_mode(self, image, common_state):
        """
        Apply same conv layer to every item in batch
        """
        x = image
        common_state = tf.reshape(common_state, [1, -1])
        stem_query = self.state_to_query(common_state)
        stem_embeddings = self.query_to_embeddings(stem_query)
        stem_embeddings = tf.split(stem_embeddings, self.embedding_splits, axis=1)
        for i, config in enumerate(self.network_config):
            fh, fw, strides, padding, fs = config
            num_out_channels = self.num_out_channels[i]
            num_in_channels = self.num_in_channels[i]

            layer_conv_embeddings = tf.reshape(stem_embeddings[i],
                                               (-1, self.out_channel_embedding_sizes[i]))

            # layer_conv_embeddings is an embedding vector for every out channel
            conv_params = self.embedding_to_params[i](layer_conv_embeddings)

            # num_output_channels[i] param vectors, now split into weights and biases
            num_weights_per_out_channel = fh * fw * num_in_channels
            w, b = tf.split(conv_params, [num_weights_per_out_channel, 1],
                            axis=1)
            # move out_channels to last dimension
            w = tf.transpose(w)
            w = tf.reshape(w, (fh, fw, num_in_channels, num_out_channels))
            b = tf.reshape(b, [-1])
            x = tf.nn.conv2d(x, w, strides=strides, padding=padding) + b

            # TODO consider performing layernorm

            for f in fs:
                x = f(x)

        x = tf.reshape(x, (-1, self.num_state_units))

        if self.integrate_initial_state:
            if self.combine_mode == "residual":
                state = (self.ln(common_state + x)
                         if self.layernorm_residual_stream
                         else common_state + self.ln(x))
            elif self.combine_mode == "gru":
                # Broadcast to one vector per batch item. (tensorflow should make this easier...)
                common_state_broadcasted = tf.broadcast_to(common_state,
                                                           tf.where([True, False],
                                                                    tf.shape(x),
                                                                    tf.shape(common_state)))
                if (self.gru_include_query_mode in ("attentional", "all")
                    and self.num_mlp_hidden_layers > 0):
                    stem_query_broadcasted =  tf.broadcast_to(stem_query,
                                                              tf.where([True, False],
                                                                       tf.shape(x),
                                                                       tf.shape(stem_query)))
                    prev_concat = tf.concat([common_state_broadcasted, x, stem_query_broadcasted], 1)
                else:
                    prev_concat = tf.concat([common_state_broadcasted, x], 1)

                z = self.sigmoid_layer_z(prev_concat)
                r = self.sigmoid_layer_r(prev_concat)
                h_tilde = self.linear_value_layer(
                    tf.concat([r * common_state, x] +
                              ([stem_query_broadcasted]
                               if self.gru_include_query_mode == "all"
                               and self.num_mlp_hidden_layers > 0
                               else []),
                              1)
                )
                state = (1 - z) * common_state + z * h_tilde
                state = self.ln(state)
            else:
                raise ValueError(self.combine_mode)
        else:
            state = self.ln(x)

        return state


class RecurrentDynamicCNN(tf.keras.Model):
    def __init__(self, num_iterations=2,
                 num_state_units=STATE_SIZE,
                 num_mlp_hidden_layers=2,
                 num_mlp_units=NUM_MLP_UNITS,
                 channels=CONV_CHANNELS,
                 out_channel_embedding_sizes=(6, 8, 8, 8),
                 static_weight_decay=STATIC_WD,
                 dynamic_weight_decay=DYNAMIC_WD,
                 num_embedding_to_params_hidden_layers=0,
                 layernorm_residual_stream=True,
                 initialize_embeddings=False,
                 embedding_initializer="glorot_uniform",
                 use_batchnorm=False,
                 reuse_norm=True,
                 combine_mode="residual",
                 integrate_initial_state=True,
                 gru_include_query_mode="attentional",
                 single_gru_mode=False,
                 classifier_num_iterations=1,
                 network_config=MNIST_CNN4_CONFIG,
                 num_data_channels=1,
                 **kwargs):
        super().__init__(**kwargs)

        self.initial_state = tf.Variable((tf.zeros(num_state_units)
                                          if not initialize_embeddings
                                          else (tf.keras.initializers.GlorotUniform
                                                if embedding_initializer == "glorot_uniform"
                                                else tf.keras.initializers.GlorotNormal)()(shape=(num_state_units,))),
                                         trainable=True)

        self.general_layer = IterativeLayer(
            num_iterations=num_iterations,
            num_state_units=num_state_units,
            num_mlp_hidden_layers=num_mlp_hidden_layers,
            num_mlp_units=num_mlp_units,
            channels=channels,
            out_channel_embedding_sizes=out_channel_embedding_sizes,
            static_weight_decay=static_weight_decay,
            dynamic_weight_decay=dynamic_weight_decay,
            num_embedding_to_params_hidden_layers=num_embedding_to_params_hidden_layers,
            layernorm_residual_stream=layernorm_residual_stream,
            initialize_embeddings=initialize_embeddings,
            embedding_initializer=embedding_initializer,
            use_batchnorm=use_batchnorm,
            reuse_norm=reuse_norm,
            combine_mode=combine_mode,
            integrate_initial_state=integrate_initial_state,
            gru_include_query_mode=gru_include_query_mode,
            network_config=network_config,
            num_data_channels=num_data_channels,
            name="general_iterative_layer")

        if classifier_num_iterations > 0:
            self.classifier_layer = IterativeLayer(
                num_iterations=classifier_num_iterations,
                num_state_units=num_state_units,
                num_mlp_hidden_layers=num_mlp_hidden_layers,
                num_mlp_units=num_mlp_units,
                channels=channels,
                out_channel_embedding_sizes=out_channel_embedding_sizes,
                static_weight_decay=static_weight_decay,
                dynamic_weight_decay=dynamic_weight_decay,
                num_embedding_to_params_hidden_layers=num_embedding_to_params_hidden_layers,
                layernorm_residual_stream=layernorm_residual_stream,
                initialize_embeddings=initialize_embeddings,
                embedding_initializer=embedding_initializer,
                use_batchnorm=use_batchnorm,
                reuse_norm=reuse_norm,
                combine_mode=combine_mode,
                integrate_initial_state=integrate_initial_state,
                gru_include_query_mode=gru_include_query_mode,
                network_config=network_config,
                num_data_channels=num_data_channels,
                name="classifier_iterative_layer")

            if combine_mode == "gru" and single_gru_mode:
                self.classifier_layer.sigmoid_layer_z = self.general_layer.sigmoid_layer_z
                self.classifier_layer.sigmoid_layer_r = self.general_layer.sigmoid_layer_r
                self.classifier_layer.linear_value_layer = self.general_layer.linear_value_layer

    def call(self, image):
        # First layer: apply same conv layer to every item in batch
        state = self.general_layer.call_stem_mode(image, self.initial_state)

        # Subsequent layers: Apply different conv layers to each item in batch.
        # Do a group convolution, arranging batch items as different channels of
        # a single image.
        image = tf.reshape(tf.transpose(image, [1, 2, 0, 3]),
                           [1, image.shape[1], image.shape[2], -1])

        state = self.general_layer(image, state)

        if hasattr(self, "classifier_layer"):
            state = self.classifier_layer(image, state)

        return state
