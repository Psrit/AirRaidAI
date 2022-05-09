import os
import shutil
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from dqn import DQN, LayerConfig

frame_shape = (250, 160, 3)
action_dim = 5

layer_configs = [
    LayerConfig(
        Conv2D,
        dict(filters=32, kernel_size=(8, 8), strides=4,
             activation="relu", name="Conv1", padding="same")
    ),
    LayerConfig(
        Conv2D,
        dict(filters=64, kernel_size=(4, 4), strides=2,
             activation="relu", name="Conv2", padding="same")
    ),
    LayerConfig(
        Conv2D,
        dict(filters=64, kernel_size=(3, 3), strides=1,
             activation="relu", name="Conv3", padding="same")
    ),
    LayerConfig(
        Flatten, {}
    ),
    LayerConfig(
        Dense,
        dict(units=512, activation="relu", name="FullyConnected1")
    ),
    LayerConfig(
        Dense,
        dict(units=action_dim, activation="relu", name="FullyConnected2")
    )
]


class DQNTest(unittest.TestCase):
    def test_output_dim(self):
        network = DQN(frame_shape, layer_configs,
                      frame_preprocessor=lambda x: x)
        network.q_model.summary()
        self.assertEqual(network.action_dim, action_dim)
        self.assertEqual(network.q_model.output_shape, (None, action_dim))

    def test_save_and_load(self):
        test_input = np.random.random((128, *frame_shape))
        test_output = np.random.random((128, action_dim))

        save_path = "./q_model_save_test/"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        network = DQN(frame_shape, layer_configs,
                      frame_preprocessor=lambda x: x)
        history: tf.keras.callbacks.History = network.q_model.fit(
            test_input, test_output
        )  # only for test; in practice use `network.perceive` to train the model
        network.loss_history += history.history["loss"]

        network.save(save_path)

        self.assertTrue(network.loss_history == [])
        self.assertTrue(
            np.array_equal(
                history.history["loss"],
                tuple(
                    v for v in np.load(save_path + "loss_records.npz").values()
                )[0]
            )
        )

        loaded_network = DQN(frame_shape, load_save_path=save_path,
                             frame_preprocessor=lambda x: x)

        self.assertTrue(loaded_network.q_model.optimizer is not None)
        self.assertTrue(loaded_network.q_model.loss is not None)
        self.assertTrue(loaded_network.q_model_target.optimizer is None)
        with self.assertRaisesRegex(
            AttributeError, "'Functional' object has no attribute 'loss'"
        ):
            _ = loaded_network.q_model_target.loss

        self.assertTrue(len(loaded_network.q_model.trainable_weights) ==
                        len(network.q_model.trainable_weights))
        # Values of trainable weights in q_model of the original network and
        # the loaded network are equal. Note that the weights in q_model_target
        # of the original network have not been synchronized yet.
        for _w1, _w2 in zip(
            loaded_network.q_model.trainable_weights,
            network.q_model.trainable_weights
        ):
            self.assertTrue(tf.reduce_all(_w1 == _w2))

        self.assertTrue(
            len(loaded_network.q_model_target.trainable_weights)
            == 0
        )

        # The reconstructed model is already compiled and has retained the
        # optimizer state, so training can resume.
        # Note that the following assertions will not hold if we replace
        # `network.q_model` with `network.q_model_target` since the weights
        # of `network.q_model_target` have not been synchronized yet.
        self.assertTrue(
            np.allclose(
                network.q_model.predict(test_input),
                loaded_network.q_model.predict(test_input)
            )
        )
        self.assertTrue(
            np.allclose(
                network.q_model.predict(test_input),
                loaded_network.q_model_target.predict(test_input)
            )
        )

        shutil.rmtree(save_path)
