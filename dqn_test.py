import os
import shutil
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from dqn import DQN, LayerConfig

state_shape = (250, 160, 3)
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
        network = DQN(state_shape, layer_configs,
                      reward_gamma=0.9, state_dtype=tf.float64)
        network.q_model.summary()
        self.assertEqual(network.action_dim, action_dim)
        self.assertEqual(network.q_model.output_shape, (None, action_dim))

    def test_save_and_load(self):
        test_states = np.random.random((128, *state_shape)).astype(np.float32)
        test_actions = np.random.randint(0, action_dim, (128,))
        test_rewards = np.random.random((128,)).astype(np.float32)
        test_dones = np.random.randint(0, 2, (128,)).astype(np.float32)
        test_states_ = np.random.random((128, *state_shape)).astype(np.float32)

        save_path = "./q_model_save_test/"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        # Create the original DQN instance and train two turns
        network = DQN(state_shape, layer_configs, reward_gamma=0.9)
        network.train(
            b_state=test_states,
            b_action=test_actions,
            b_reward=test_rewards,
            b_done=test_dones,
            b_state_=test_states_
        )
        network.train(
            b_state=test_states,
            b_action=test_actions,
            b_reward=test_rewards,
            b_done=test_dones,
            b_state_=test_states_
        )
        loss_history = network.loss_history

        # Save the instance
        network.save(save_path)

        # Loss records is saved correctly
        self.assertTrue(network.loss_history == [])
        self.assertTrue(
            np.array_equal(
                loss_history,
                tuple(
                    v for v in np.load(save_path + "loss_records.npz").values()
                )[0]
            )
        )

        # Load the instance
        loaded_network = DQN(
            state_shape, layer_configs=layer_configs
        )
        loaded_network.load(save_path)

        # Weights of Q models are equal
        self.assertEqual(len(loaded_network.q_model.trainable_weights),
                         len(network.q_model.trainable_weights))
        # Values of trainable weights in q_model of the original network and
        # the loaded network are equal. Note that the weights in q_model_target
        # of the original network have not been synchronized yet.
        for _w1, _w2 in zip(
            loaded_network.q_model.trainable_weights,
            network.q_model.trainable_weights
        ):
            self.assertTrue(tf.reduce_all(_w1 == _w2))

        # Loaded target Q model is not trainable
        self.assertTrue(
            len(loaded_network.q_model_target.trainable_weights)
            == 0
        )

        # Optimizers are equal
        self.assertEqual(network.optimizer, loaded_network.optimizer)
        self.assertEqual(len(loaded_network.optimizer.variables()),
                         len(network.optimizer.variables()))
        for opvar, opvar_tar in zip(
            network.optimizer.variables(), loaded_network.optimizer.variables()
        ):
            self.assertTrue(tf.reduce_all(opvar == opvar_tar))

        # Hyperparameters are equal
        self.assertEqual(
            network.reward_gamma, loaded_network.reward_gamma
        )
        self.assertEqual(
            network.epsilon(network.num_trained_steps),
            loaded_network.epsilon(loaded_network.num_trained_steps)
        )
        self.assertEqual(
            network.train_steps_per_q_sync, loaded_network.train_steps_per_q_sync
        )

        # Outputs of original and loaded (target) Q networks are identical.
        # In following comments, prime (') indicates the loaded network.
        # Q(s) == Q_{target}(s) may not hold, since they may have not been
        # synchronized yet. Same for Q'(s) and Q'_{target}(s).
        # Q(s) == Q'(s)
        self.assertTrue(
            np.allclose(
                network.q_model.predict(test_states),
                loaded_network.q_model.predict(test_states)
            )
        )
        # Q_{target}(s) == Q'_{target}(s)
        self.assertTrue(
            np.allclose(
                network.q_model_target.predict(test_states),
                loaded_network.q_model_target.predict(test_states)
            )
        )

        # loaded DQN can also be trained
        loaded_network.train(
            b_state=test_states,
            b_action=test_actions,
            b_reward=test_rewards,
            b_done=test_dones,
            b_state_=test_states_
        )

        shutil.rmtree(save_path)
