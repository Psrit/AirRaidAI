from __future__ import print_function
import random

import tensorflow as tf
import numpy as np
import os

from collections import deque

INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
REPLAY_SIZE = 320
BATCH_SIZE = 64
GAMMA = 0.9

SAVE_STEP = 1000  # Auto save once after training if time_step % SAVE_STEP == 0.
TRAIN_STEP = 100  # Train once after perceiving if time_step % TRAIN_STEP == 0.


class DQNBrain:
    """
    A DQN network with two convolution layers and two fully connected layers.

    To use this network, please follow the steps:

    1. initialize DQNBrain
    brain = DQNBrain(env, 4)  # , [8, 4], [16, 32], 256, 4)
    
    2. add network layers
    brain.add_conv("conv1", 4, 16, pooling=True)
    brain.add_conv("conv2", 4, 32, pooling=True)
    brain.add_fc("fc1", 256)
    brain.add_fc("q_layer", -1, output_layer=True)

    These will set the network made up of some convolution layers firstly and 
    then fully connected layers. Note that the last layer added must be a fully
    connected layer whose boolean flag `output_layer` must be True, in which 
    case `num_nodes` will be set by `add_fc` (and therefore you can use any 
    value).

    3. initialize Q network
    brain.initialize_network()

    """

    def __init__(self, env, pooling_scale, record=False,
                 save_path='saved_networks', record_path="records"):
        """
        Initializes the DQN network.
        
        :param env: OpenAI Gym env
            Gives the game environment. The shape of the observation space of
            `env` must be a 3-tuple, e.g. (width of the image, height of the
            image, number of channels).
        :param pooling_scale: int
            Gives the pooling scale in the max_pool steps (if any).
        :param record: boolean
            Tells the network whether the costs is needed to be recorded.
        :param save_path: str
            Gives the path where the network is supposed to be reloaded from 
            and saved to.
        :param record_path: str
            Gives the path where the cost values are supposed to be recorded.

        """
        # init replay memory
        self.replay_memory = deque()

        # hyper parameters for DQN algorithm
        self.time_step = 0
        self.train_step = 0
        self.epsilon = INITIAL_EPSILON
        self.pooling_scale = pooling_scale

        # env-related parameters
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n

        # layers
        self.state_input = None
        self.q_value = None

        # other attributes in the network
        self.cost = None
        self.optimizer = None
        self.y = None
        self.action_input = None

        # DEBUGGER
        self.hidden_record = None
        self.fc1_record = None

        # hyper parameters for the network
        self.conv_config = []  # list of dict
        self.fc_config = []  # list of dict
        self.conv_weights = []
        self.conv_biases = []
        self.fc_weigths = []
        self.fc_biases = []

        # cheat parameters:
        # The last conv layer's shape (width, height, depth);
        # batch_size is not included here
        self.last_conv_layer_shape = np.array(self.state_dim)
        self.output_layer_implemented = False

        self.session = None
        self.saver = None

        if not save_path.endswith(os.sep):
            save_path += os.sep
        self.save_path = save_path

        if not record_path.endswith(os.sep):
            record_path += os.sep
        self.record_path = record_path

        self.record = record

    def define_inputs(self):
        with tf.name_scope('inputs'):
            self.state_input = \
                tf.placeholder('float', [None] + list(self.state_dim))

    def add_conv(self, name, patch_size, depth, activation="sigmoid", pooling=False):
        in_depth = self.state_dim[-1] if len(self.conv_config) == 0 \
            else self.conv_config[-1]['out_depth']
        self.conv_config.append({
            'patch_size': patch_size,
            'in_depth': in_depth,
            'out_depth': depth,
            'activation': activation,
            'pooling': pooling,
            'name': name
        })
        self.last_conv_layer_shape[-1] = depth
        if pooling is True:
            old_shape = self.last_conv_layer_shape
            scale = np.array([self.pooling_scale, self.pooling_scale, 1])
            self.last_conv_layer_shape = \
                np.ceil(1.0 * old_shape / scale).astype(int)
        with tf.name_scope(name):
            weights = self.weight_variable(
                [patch_size, patch_size, in_depth, depth], name
            )
            biases = self.bias_variable([depth], name)
            self.conv_weights.append(weights)
            self.conv_biases.append(biases)

    def add_fc(self, name, num_nodes, activation="sigmoid", output_layer=False):
        if self.output_layer_implemented:
            print("The output layer has been implemented. "
                  "This layer will be ignored.")
            return

        # set num_nodes
        if output_layer is True:
            num_nodes = self.action_dim
            self.output_layer_implemented = True

        # set in_num_nodes
        in_num_nodes = \
            np.prod(self.last_conv_layer_shape) if len(self.fc_config) == 0 \
                else self.fc_config[-1]['out_num_nodes']

        self.fc_config.append({
            'in_num_nodes': in_num_nodes,
            'out_num_nodes': num_nodes,
            'activation': activation,
            'name': name,
            'output_layer': output_layer
        })
        with tf.name_scope(name):
            weights = self.weight_variable([in_num_nodes, num_nodes], name)
            biases = self.bias_variable([num_nodes], name)
            self.fc_weigths.append(weights)
            self.fc_biases.append(biases)
            # self.train_summaries.append(tf.histogram_summary(str(len(self.fc_weights)) + '_weights', weights))
            # self.train_summaries.append(tf.histogram_summary(str(len(self.fc_biases)) + '_biases', biases))

    def create_q_network(self):
        """
        Define variables in the network.

        """

        # ------------- hidden layers -------------
        def calculate_q_layer(data):
            """
            Calculates those hidden layers, and returns the Q layer.

            """
            # -------------------- convolution layers ---------------------
            for i, (weights, biases, config) in enumerate(
                    zip(self.conv_weights, self.conv_biases, self.conv_config)
            ):
                with tf.name_scope(config['name'] + '_model'):
                    with tf.name_scope('convolution'):
                        data = tf.nn.conv2d(
                            data, filter=weights, strides=[1, 1, 1, 1],
                            padding='SAME'
                        )
                        data = data + biases

                if config['activation'] == 'sigmoid':
                    data = tf.nn.sigmoid(data)
                elif config['activation'] is None:
                    pass
                else:
                    print('Activation function can only be sigmoid or relu now.')
                    data = tf.nn.relu(data)

                if config['pooling']:
                    data = self.max_pool_nxn(data, self.pooling_scale)

            # ------------------ fully connected layers -------------------
            for i, (weights, biases, config) in enumerate(
                    zip(self.fc_weigths, self.fc_biases, self.fc_config)
            ):
                print(weights)
                if i == 0:
                    shape = data.get_shape().as_list()
                    data = tf.reshape(data, [-1, np.prod(shape[1:])])

                with tf.name_scope(config['name'] + 'model'):
                    data = tf.matmul(data, weights) + biases

                    if config['activation'] == 'sigmoid':
                        data = tf.nn.sigmoid(data)
                    elif config['activation'] == None:
                        pass
                    else:
                        print('Activation function can only be sigmoid or relu now.')
                        data = tf.nn.relu(data)

            # ----------- last fully connected layer (Q layer) ------------
            # return.shape = [batch_size, action_dim]
            return data

        # Q value layer
        # shape =  [batch_size, action_dim]
        self.q_value = calculate_q_layer(self.state_input)

    def create_training_method(self):
        self.action_input = tf.placeholder('float', [None, self.action_dim])
        self.y = tf.placeholder('float', [None])
        q_action = tf.reduce_sum(tf.multiply(self.q_value, self.action_input), axis=1)

        with tf.name_scope('loss'):
            self.cost = tf.reduce_mean(tf.square(self.y - q_action))

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.cost)

    def initialize_network(self):
        """
        Initializes the network. Only being called after all layers implemented
        is allowed.

        """
        if self.output_layer_implemented is False:
            raise Exception("The output layer is not implemented yet!")

        self.define_inputs()

        self.create_q_network()
        self.create_training_method()

        # init saver
        self.saver = tf.train.Saver(tf.all_variables())

        # init session
        self.session = tf.InteractiveSession()

        self.session.run(tf.global_variables_initializer())

        # load checkpoints if there are any
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir=self.save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("[Load checkpoint] Successfully loaded: ", checkpoint.model_checkpoint_path)
        else:
            print("[Load checkpoint] Could not find saved network weights. Start from scratch.")

    def perceive(self, state, action, reward, next_state, done):
        """
        Let the brain perceive the state transition relation:

            state --action--> reward, next_state, done

        :param state:
            Former state.
        :param action: int
            Action that yields `next_state` from `state`.
        :param reward:
            Reward of the action.
        :param next_state:
            Next state.
        :param done:
            Tells whether `next_state` is a terminal state.

        """
        self.time_step += 1

        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_memory.append(
            (state, one_hot_action, reward, next_state, done)
        )

        if len(self.replay_memory) > REPLAY_SIZE:
            self.replay_memory.popleft()

        if len(self.replay_memory) > BATCH_SIZE and self.time_step % TRAIN_STEP == 0:
            self.train_q_network()

    def train_q_network(self):
        print("[Training round {0}] ready to train...".format(self.train_step))

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        next_q_value_batch = self.q_value.eval(
            feed_dict={self.state_input: next_state_batch}
        )
        for i in range(BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] +
                               GAMMA * np.max(next_q_value_batch[i]))

        self.optimizer.run(
            feed_dict={
                self.y: y_batch,
                self.action_input: action_batch,
                self.state_input: state_batch
            }
        )
        cost_print = self.cost.eval(
            feed_dict={
                self.y: y_batch,
                self.action_input: action_batch,
                self.state_input: state_batch
            }
        )
        # conv1_w_print = self.conv1_weights.eval(
        #     feed_dict={self.state_input: state_batch}
        # )
        # print("State batch: {0}".format(state_batch))
        # print("Next q batch: {0}".format(next_q_value_batch))
        # print("reward batch: {0}".format(reward_batch))
        # print("y: {0}".format(y_batch))
        # print("fc1_record: {0}".format(self.fc1_record.eval(
        #     feed_dict={self.state_input: next_state_batch}
        # )))
        # print("hidden: {0}".format(self.hidden_record.eval(
        #     feed_dict={self.state_input: next_state_batch}
        # )))
        # print("fc1_weights: {0}".format(self.session.run(
        #     self.fc1_weights, 
        #     feed_dict={self.state_input: next_state_batch}
        # )))
        print("network time step {0} | cost: "
              .format(self.time_step), cost_print)

        if self.record:
            # save cost value
            if not os.path.isdir(self.record_path):
                os.makedirs(self.record_path)
            record_file = self.record_path + "costs"
            rf = open(record_file, "a")
            try:
                rf.write(str(cost_print) + "\n")
            finally:
                rf.close()

        # auto-save
        if self.time_step % SAVE_STEP == 0:
            if os.path.isdir(self.save_path):
                saved_filename = \
                    self.saver.save(self.session, self.save_path + 'model.ckpt',
                                    global_step=self.time_step)
                print("[AUTO SAVE] Model saved in file: {0}"
                      .format(saved_filename))
            else:
                os.makedirs(self.save_path.split('/')[0])
                saved_filename = \
                    self.saver.save(self.session, self.save_path + 'model.ckpt',
                                    global_step=self.time_step)
                print("[AUTO SAVE] Model saved in file: {0}"
                      .format(saved_filename))

        self.train_step += 1

    def epsilon_greedy_action(self, state):
        """
        Given current state, selects the next action.
        
        :param state:
            Current state. Note that the state here is actually a single 
            observation of self.env, not a batch of states.
        :return: 
            Next action.

        """
        # Here batch_size = 1, so self.q_value.eval(...).shape =
        # [1, action_dim].
        q_value = self.q_value.eval(
            feed_dict={self.state_input: [state]}
        )[0]  # it would be OK if there is no "[0]".

        if random.random() <= self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            action = np.argmax(q_value)

        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000.0

        return action

    def action(self, state):
        return np.argmax(self.q_value.eval(
            feed_dict={self.state_input: [state]}
        ))

    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.5)
        return tf.Variable(initial, name=name + '_weights')

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial, name=name + '_biases')

    @staticmethod
    def max_pool_nxn(data, n):
        return tf.nn.max_pool(data, ksize=[1, n, n, 1],
                              strides=[1, n, n, 1], padding="SAME")
