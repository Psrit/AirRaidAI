import unittest

import numpy as np

from replaymemory import ObsvTransition, ReplayMemory


class ReplayMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.obsv_shape = (2, 3)
        self.n_actions = 5
        return super().setUp()

    def fill_memory(self, memory: ReplayMemory, counts=None):
        if counts is None:
            counts = memory.capacity
        for i in range(0, counts):
            memory.add(
                ObsvTransition(
                    observation=np.random.rand(*self.obsv_shape),
                    action=np.random.randint(0, self.n_actions),
                    reward=i,
                    done=True,
                    observation_=None  # just for test
                )
            )

    def test_is_valid_index(self):
        memory1 = ReplayMemory(
            observation_shape=self.obsv_shape,
            capacity=10,
            hist_len=2,
            hist_type="linear",
            hist_spacing=2  # hist_index_shifts = [-2, 0]
        )
        self.fill_memory(memory1)
        memory1.dones[5] = False
        memory1.dones[7] = False
        print("[memory1]")
        print(memory1.info())
        print(memory1.dones)
        print(memory1.hist_index_shifts)
        self.assertTrue(
            np.all([
                not memory1.is_valid_index(
                    index, allow_terminate_at_index=False)
                for index in range(memory1.capacity)
            ])
        )

        memory1.dones[6] = False
        self.assertTrue(
            np.all([
                memory1.is_valid_index(index, allow_terminate_at_index=False)
                == (True if index == 7 else False)
                for index in range(memory1.capacity)
            ])
        )

        memory2 = ReplayMemory(
            observation_shape=self.obsv_shape,
            capacity=10,
            hist_len=2,
            hist_type="linear",
            hist_spacing=1  # hist_index_shifts = [-1, 0]
        )
        self.fill_memory(memory2)
        memory2.dones[7] = False
        memory2.dones[8] = False
        print("[memory2]")
        print(memory2.info())
        print(memory2.dones)
        print(memory2.hist_index_shifts)
        self.assertTrue(
            np.all([
                memory2.is_valid_index(index, allow_terminate_at_index=False)
                == (True if (index == 8) else False)
                for index in range(memory2.capacity)
            ])
        )

    def test_sample(self):
        memory1 = ReplayMemory(
            observation_shape=self.obsv_shape,
            capacity=10,
            hist_len=2,
            hist_type="linear",
            hist_spacing=2  # hist_index_shifts = [-2, 0]
        )
        self.fill_memory(memory1)
        memory1.dones[7] = False
        print("[memory1]")
        print(memory1.info())
        print(memory1.dones)

        with self.assertRaises(RuntimeError):
            memory1.sample(100)

        memory2 = ReplayMemory(
            observation_shape=self.obsv_shape,
            capacity=20,
            hist_len=3,
            hist_type="linear",
            hist_spacing=2  # hist_index_shifts = [-4, -2, 0]
        )
        self.fill_memory(memory2, counts=13)
        memory2.dones[4] = False
        memory2.dones[6] = False
        memory2.dones[7] = False
        memory2.dones[8] = False
        memory2.dones[9] = False
        memory2.dones[11] = False
        memory2.dones[13] = False
        memory2.dones[15] = False
        print("[memory2]")
        print(memory2.info())
        print(memory2.dones)
        self.assertTrue(np.array_equal(
            memory2.sampleable_range(),
            np.arange(4, 16)
        ))
        self.assertTrue(
            np.all([
                memory2.is_valid_index(index, allow_terminate_at_index=False) ==
                (True if (index in (4,)) else False)
                for index in range(memory2.capacity)
            ])
        )

        sampled_counts = len(memory2.sample(50))
        print(f"Sampled counts = {sampled_counts}")
        self.assertTrue(sampled_counts > 0)

    def test_sample_range_raise(self):
        memory = ReplayMemory(
            observation_shape=self.obsv_shape,
            capacity=10,
            hist_len=3,
            hist_type="linear",
            hist_spacing=2  # hist_index_shifts = [-4, -2, 0]
        )
        self.fill_memory(memory=memory, counts=1)
        self.assertEqual(memory.size, 1 - memory.hist_index_shifts[0])
        with self.assertRaises(RuntimeError):
            print(memory.sampleable_range(raise_for_empty_range=True))

    def test_construct_current_state(self):
        memory = ReplayMemory(
            observation_shape=self.obsv_shape,
            capacity=10,
            hist_len=3,
            hist_type="linear",
            hist_spacing=3  # hist_index_shifts = [-6, -3, 0]
        )
        self.assertTrue(np.array_equal(
            memory.construct_current_state(
                np.random.rand(*self.obsv_shape)
            )[:-1],
            np.zeros((2, *self.obsv_shape))
        ))

        for i in range(0, 11):
            memory.add(ObsvTransition(
                np.ones(self.obsv_shape) * i,
                action=np.random.randint(self.n_actions),
                reward=i,
                done=False
            ))
        memory.dones[2] = True
        print(memory.info())
        print(memory.rewards)
        print(memory.dones)
        self.assertTrue(np.array_equal(
            memory.construct_current_state(
                np.random.rand(*self.obsv_shape)
            )[:-1],
            np.array([
                np.ones(self.obsv_shape) * 0,
                np.ones(self.obsv_shape) * 8
            ])
        ))

    def test_construct_current_state_shortest_hist_len(self):
        memory = ReplayMemory(
            observation_shape=self.obsv_shape,
            capacity=10,
            hist_len=1,
            hist_type="linear",
            # `hist_spacing` can be any value since hist_index_shifts = [0]
            hist_spacing=3
        )
        frame = np.random.rand(*self.obsv_shape)
        self.assertTrue(np.array_equal(
            memory.construct_current_state(frame),
            [frame.astype(memory.dtype)]
        ))
