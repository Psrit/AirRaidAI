import os
import shutil
import time
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten

from dqn import LayerConfig, create_q_model

state_shape = (3, 2, 4)
action_dim = 3
layer_configs = (
    LayerConfig(
        Dense, dict(units=5)
    ),
    LayerConfig(
        Flatten, {}
    ),
    LayerConfig(
        Dense, dict(units=action_dim)
    )
)

batch_size = 128
test_input = np.random.random((batch_size, *state_shape))
test_output = np.random.random((batch_size, action_dim))

save_path = "./q_model_save_test"


class CreatQModelTest(unittest.TestCase):
    def setUp(self) -> None:
        optimizer = tf.keras.optimizers.RMSprop()
        loss = tf.keras.losses.mse

        self.q_model = create_q_model(
            state_shape=state_shape,
            layer_configs=layer_configs,
            is_target_network=False
        )
        self.q_model.compile(
            optimizer=optimizer, loss=loss
        )

        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists(save_path):
            if os.path.isdir(save_path):
                shutil.rmtree(save_path)
            else:
                os.remove(save_path)
        return super().tearDown()

    def test_save_and_load(self):
        self.q_model.fit(test_input, test_output, batch_size=4)
        # For TensorFlow with version <= 2.7.0, if saving model as "tf" format,
        # the optimizer status cannot be recovered by `load_model`;
        # see https://github.com/keras-team/keras/issues/15512.
        print("Save 1")
        self.q_model.save(save_path, save_format="h5")
        time.sleep(3)

        self.q_model.fit(test_input, test_output, batch_size=4)
        # Cover the previous record
        print("Save 2")
        self.q_model.save(save_path, save_format="h5")
        time.sleep(3)

        reconstructed_q_model: tf.keras.Model = tf.keras.models.load_model(
            save_path
        )
        # The reconstructed model is already compiled and has retained the optimizer
        # state, so training can resume:
        self.assertTrue(
            np.all(
                np.array_equal(w1, w2)
                for w1, w2 in zip(
                    self.q_model.optimizer.variables,
                    reconstructed_q_model.optimizer.variables
                )
            )
        )
        self.assertTrue(
            np.allclose(
                self.q_model.predict(test_input),
                reconstructed_q_model.predict(test_input)
            )
        )

    def test_load_trainable_and_target_models(self):
        self.q_model.fit(test_input, test_output, batch_size=4)
        # For TensorFlow with version <= 2.7.0, if saving model as "tf" format,
        # the optimizer status cannot be recovered by `load_model`;
        # see https://github.com/keras-team/keras/issues/15512.
        self.q_model.save(save_path, save_format="h5")

        reconstructed_q_model: tf.keras.Model = tf.keras.models.load_model(
            save_path, compile=True
        )
        reconstructed_q_model.trainable = True

        reconstructed_q_model_target: tf.keras.Model = tf.keras.models.load_model(
            save_path, compile=False
        )
        reconstructed_q_model_target.trainable = False

        self.assertTrue(reconstructed_q_model.optimizer is not None)
        self.assertTrue(reconstructed_q_model.loss is not None)
        self.assertTrue(reconstructed_q_model_target.optimizer is None)
        with self.assertRaisesRegex(
            AttributeError, "'Functional' object has no attribute 'loss'"
        ):
            _ = reconstructed_q_model_target.loss

        self.assertTrue(len(reconstructed_q_model.trainable_weights) ==
                        len(self.q_model.trainable_weights))
        for _w1, _w2 in zip(
            reconstructed_q_model.trainable_weights,
            self.q_model.trainable_weights
        ):
            self.assertTrue(tf.reduce_all(_w1 == _w2))
        self.assertTrue(len(reconstructed_q_model_target.trainable_weights) ==
                        0)

        # The reconstructed model is already compiled and has retained the optimizer
        # state, so training can resume:
        self.assertTrue(
            np.allclose(
                self.q_model.predict(test_input),
                reconstructed_q_model.predict(test_input)
            )
        )
        self.assertTrue(
            np.allclose(
                self.q_model.predict(test_input),
                reconstructed_q_model_target.predict(test_input)
            )
        )
