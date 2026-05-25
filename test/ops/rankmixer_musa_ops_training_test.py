# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
# ==============================================================================

"""Lightweight RankMixer training test using tensorflow_musa.ops."""

import numpy as np
import tensorflow as tf

from musa_test_utils import MUSATestCase
from tensorflow_musa import ops as musa_ops


class MusaLayerNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.gamma = self.add_weight(
            "gamma", shape=[dim], initializer=tf.ones_initializer()
        )
        self.beta = self.add_weight(
            "beta", shape=[dim], initializer=tf.zeros_initializer()
        )

    def call(self, x):
        return musa_ops.layer_norm(x, self.gamma, self.beta, epsilon=self.epsilon)


class SemanticTokenization(tf.keras.layers.Layer):
    def __init__(self, num_tokens, dim_emb):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim_emb = dim_emb
        self.dense_layers = [
            tf.keras.layers.Dense(dim_emb, activation="linear")
            for _ in range(num_tokens)
        ]

    def call(self, x):
        chunks = tf.split(x, self.num_tokens, axis=-1)
        outputs = [layer(chunk) for chunk, layer in zip(chunks, self.dense_layers)]
        return tf.stack(outputs, axis=1)


class TokenMixer(tf.keras.layers.Layer):
    def __init__(self, num_tokens, dim_emb, num_heads):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        self.d_k = dim_emb // num_heads

    def call(self, x):
        x = tf.reshape(x, (-1, self.num_tokens, self.num_heads, self.d_k))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (-1, self.num_heads, self.num_tokens * self.d_k))


class PerTokenFFN(tf.keras.layers.Layer):
    def __init__(self, num_tokens, dim_emb, expansion_ratio=2, dropout_rate=0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.experts = []
        for i in range(num_tokens):
            self.experts.append(
                [
                    tf.keras.layers.Dense(dim_emb * expansion_ratio, name=f"expert_{i}_fc1"),
                    tf.keras.layers.Dense(dim_emb, name=f"expert_{i}_fc2"),
                ]
            )

    def call(self, x, training=False):
        outputs = []
        for i, expert_layers in enumerate(self.experts):
            h = x[:, i, :]
            h = expert_layers[0](h)
            h = musa_ops.gelu(h, approximate=True)
            if training and self.dropout_rate > 0.0:
                h, _ = musa_ops.dropout(
                    h,
                    rate=self.dropout_rate,
                    seed=1234 + i,
                    offset=0,
                )
            h = expert_layers[1](h)
            outputs.append(h)
        return tf.stack(outputs, axis=1)


class RankMixerLayer(tf.keras.layers.Layer):
    def __init__(self, num_tokens, dim_emb, num_heads, expansion_ratio, dropout_rate=0.0):
        super().__init__()
        self.token_mixer = TokenMixer(num_tokens, dim_emb, num_heads)
        self.per_token_ffn = PerTokenFFN(
            num_tokens, dim_emb, expansion_ratio, dropout_rate=dropout_rate
        )
        self.norm1 = MusaLayerNorm(epsilon=1e-5)
        self.norm2 = MusaLayerNorm(epsilon=1e-5)

    def call(self, x, training=False):
        mixed_x = self.token_mixer(x)
        x = self.norm1(x + mixed_x)
        return self.norm2(x + self.per_token_ffn(x, training=training))


class RankMixer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_sparse_embs,
        num_tokens=4,
        dim_input_dense=3,
        dim_emb=16,
        num_layers=2,
        expansion_ratio=2,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.num_sparse_embs = num_sparse_embs
        self.num_tokens = num_tokens
        self.dim_input_dense = dim_input_dense
        self.dim_input_sparse = len(num_sparse_embs)
        self.dim_emb = dim_emb
        self.sparse_embeddings = [
            tf.keras.layers.Embedding(input_dim=n, output_dim=dim_emb)
            for n in num_sparse_embs
        ]
        self.dense_embedding = tf.keras.layers.Dense(dim_input_dense * dim_emb)
        self.semantic_tokenization = SemanticTokenization(num_tokens, dim_emb)
        self.layers_list = [
            RankMixerLayer(
                num_tokens,
                dim_emb,
                num_tokens,
                expansion_ratio,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ]
        self.projection_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, inputs, training=False):
        sparse_inputs, dense_inputs = inputs
        sparse_outputs = [
            embedding(sparse_inputs[:, i])
            for i, embedding in enumerate(self.sparse_embeddings)
        ]
        sparse_outputs = tf.stack(sparse_outputs, axis=1)
        dense_outputs = self.dense_embedding(dense_inputs)
        dense_outputs = tf.reshape(
            dense_outputs, [-1, self.dim_input_dense, self.dim_emb]
        )
        x = tf.concat((sparse_outputs, dense_outputs), axis=1)
        x = tf.reshape(
            x,
            (-1, (self.dim_input_dense + self.dim_input_sparse) * self.dim_emb),
        )
        x = self.semantic_tokenization(x)
        for layer in self.layers_list:
            x = layer(x, training=training)
        x = tf.reduce_mean(x, axis=1)
        return self.projection_head(x)


class RankMixerMusaOpsTrainingTest(MUSATestCase):
    def testRankMixerTrainingWithMusaOpsConverges(self):
        np.random.seed(123)
        tf.random.set_seed(123)

        num_sparse_embs = [32, 32, 32, 32, 32]
        batch_size = 64
        steps = 200
        learning_rate = 0.01

        sparse_np = np.random.randint(
            0, 32, size=(batch_size, len(num_sparse_embs)), dtype=np.int32
        )
        dense_np = np.random.normal(size=(batch_size, 3)).astype(np.float32)
        labels_np = (
            (sparse_np[:, 0] % 2).astype(np.float32)
            + 0.5 * dense_np[:, 0]
            - 0.25 * dense_np[:, 1]
            > 0.35
        ).astype(np.float32)

        with tf.device("/device:MUSA:0"):
            model = RankMixer(num_sparse_embs=num_sparse_embs)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            sparse_inputs = tf.constant(sparse_np)
            dense_inputs = tf.constant(dense_np)
            labels = tf.constant(labels_np)

            losses = []
            for _ in range(steps):
                with tf.GradientTape() as tape:
                    logits = tf.squeeze(model((sparse_inputs, dense_inputs), training=True))
                    loss = criterion(labels, logits)
                grads = tape.gradient(loss, model.trainable_variables)
                self.assertTrue(any(grad is not None for grad in grads))
                optimizer.apply_gradients(
                    [
                        (grad, var)
                        for grad, var in zip(grads, model.trainable_variables)
                        if grad is not None
                    ]
                )
                losses.append(float(loss.numpy()))

        self.assertLess(losses[-1], losses[0] * 0.5)
        self.assertLess(losses[-1], 0.2)


if __name__ == "__main__":
    tf.test.main()
