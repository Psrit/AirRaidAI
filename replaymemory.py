import numbers
import typing

import numpy as np


class FrameTransition(typing.NamedTuple):
    frame: typing.Any = None
    action: numbers.Integral = None
    reward: float = None
    done: bool = None


class ReplayMemory(object):
    def __init__(
            self,
            frame_shape: typing.Tuple[int, ...],
            capacity: int,
            hist_len: int = 1,
            hist_type: str = "linear",
            hist_spacing: int = 1,
            max_sample_attempts: int = 1000,
            dtype=np.float32
    ) -> None:
        """
        Initialize the replay memory.

        :param frame_shape:
            The shape of each frame.
            Note that if raw frames from the game are to be preprocessed,
            it should be handled by the memory maintainer and the frames
            stored here are already preprocessed.
        :param capacity:
            Maximum size of the memory.
        :param hist_len:
            The length of the historical sequence that defines a state.
            The state of time t is composed of a sequence of historical frames
            (e.g. game screen images). If the memory contains frames as
                [f0, f1, ..., fn],
            then the state s_i is
                [  f_{i+hist_index_shifts[0]}, ...,
                   f_{i+hist_index_shifts[j]}, ...,
                   f_{i+hist_index_shifts[hist_len-2]},
                   f_{i}  ].
            Note that:
            (i) hist_index_shifts[-1] is always 0;
            (ii) capacity - 1 + hist_index_shifts[0] >= 1.  (*)
            We take `capacity` as a more solid restriction, and use (*) to adjust
            the actual `hist_len` if needed.
        :param hist_type:
            Defines which frames influcence the state at a certain moment.
            Available options are "linear", "exp2", "exp1.5".
            Exponential types imply that more previous frames have less
            influences on current state.
        :param hist_spacing:
            Defines the "linear" historical sequence type.
        :param max_sample_attempts:
            The maximum number of attempts allowed to get a sample.
        :param dtype:
            Data type for recorded preprocessed frames.

        """
        if not capacity > 0:
            raise ValueError(f"Invalid capacity: {capacity}")
        self.capacity = capacity

        self.frame_shape = frame_shape

        if hist_type == "linear":
            self.hist_index_shifts = np.arange(0, hist_len, 1) * hist_spacing
        elif hist_type == "exp2":
            self.hist_index_shifts = np.power(
                2, np.arange(0, hist_len, 1)
            ).astype(int) - 1
        elif hist_type == "exp1.5":
            self.hist_index_shifts = np.ceil(
                np.power(1.5, np.arange(0, hist_len, 1))
            ).astype(int) - 1

        self.hist_index_shifts = self.hist_index_shifts[
            self.hist_index_shifts < self.capacity - 1
        ]
        self.hist_index_shifts = -self.hist_index_shifts[::-1]
        self.hist_len = len(self.hist_index_shifts)

        self.dtype = dtype

        # Initialization
        self.reset()

        self._max_sample_attempts = max_sample_attempts

    def reset(self):
        # Initialize memory arrays
        self.frames = np.zeros(
            shape=(self.capacity, *self.frame_shape),
            dtype=self.dtype
        )
        self.actions = np.zeros(shape=(self.capacity,), dtype=int)
        self.rewards = np.zeros(shape=(self.capacity,), dtype=float)
        self.dones = np.zeros(shape=(self.capacity,), dtype=bool)

        # Flag and cursors indicate the range of indices of records in the
        # memory arrays
        self.is_full = False
        self._start = 0
        self._end = 0

    @property
    def size(self):
        if self._start < self._end:
            # in this case the memory is not full
            return self._end - self._start
        elif self._end < self._start:
            return self._end + self.capacity - self._start
        elif self.is_full:
            return self.capacity
        else:  # _start == _end and not is_full
            return 0  # empty

    def add(self, transition: FrameTransition) -> None:
        if self.size == 0:
            self._end = -self.hist_index_shifts[0]

        self.frames[self._end] = np.asarray(
            transition.frame, dtype=self.dtype
        )
        self.actions[self._end] = transition.action
        self.rewards[self._end] = transition.reward
        self.dones[self._end] = transition.done

        if not self.is_full:
            self._end = (self._end + 1) % self.capacity
            if self._end == self._start:
                self.is_full = True
        else:  # _start == _end
            self._start = (self._start + 1) % self.capacity
            self._end = (self._end + 1) % self.capacity

    def is_valid_index(self, index, allow_terminate_at_index=True) -> bool:
        """
        Checks whether an index is valid.

        For an index to be valid, two requirements must be satisfied:
        1. All indices of the frames in the history sequence tracing back
           from `index` must be in the wrapping range of [_start, _end).
        2. All intermediate frames in the history sequence tracing back
           from `index` (including those frames not used to construct
           `state[index]`) must be non-terminal.

        :param index:
            The index to be checked.
        :param allow_terminate_at_index:
            Tells whether the frame at `index` can be terminal (when checking
            requirement 2).

        """
        # Indices of frames that compose the history sequence of state[index].
        # Note that some elements may be negative.
        hist_indices = index + self.hist_index_shifts

        # index out of range
        if (
            index < self._end and
            np.any(hist_indices < self._end - self.size)
        ):  # index lies in [0, self._end)
            return False
        elif (
            self._end <= index and
            np.any(hist_indices < self._end + self.capacity - self.size)
        ):  # index lies in [self._end, self.capacity)
            return False

        # If there are terminal flags in intermediate frames, the history
        # sequence is not valid:
        hist_indices = hist_indices % self.capacity
        full_hist_stop_index = (
            index if allow_terminate_at_index else index + 1
        )
        if hist_indices[0] <= index:
            full_hist_indices = np.arange(
                hist_indices[0], full_hist_stop_index
            )
        else:
            full_hist_indices = np.concatenate(
                [np.arange(hist_indices[0], self.capacity),
                 np.arange(0, full_hist_stop_index)]
            )
        done_of_full_hist = self.dones[
            full_hist_indices
        ]

        if np.any(done_of_full_hist):
            return False

        return True

    def get_state(self, index):
        """
        Construct the state corresponds to a valid `index` as
            [ pframe[index + hist_index_shifts[0]], ...
              pframe[index + hist_index_shifts[i]], ...
              pframe[index + hist_index_shifts[hist_len - 2]],
              pframe[index] ]

        :param index:
            The index the state corresponds to, which is supposed to be valid
            (the caller should check by calling
                `self.is_valid_index(index, allow_terminate_at_index=True)`).
        :return:
            `state[index]`.

        """
        # assert self.is_valid_index(index, allow_terminate_at_index=True)
        return self.frames[(index + self.hist_index_shifts) % self.capacity]

    def construct_current_state(self, current_frame):
        """
        Construct the current state.

        :param current_frame:
            Current frame (which has been preprocessed if needed).
            If added in the memory, its index should be `self._end`.
        :return:
            Current state.
            Note that different from `self.get_state`, this method do not
            require all `hist_index_shifts[0]-1` frames tracing back from
            `self._end` (not included) to be non-terminal.

        """
        current_frame = np.asarray(current_frame, dtype=self.dtype)

        indices = self._end + self.hist_index_shifts[:-1]
        if len(indices) == 0:  # i.e. hist_len == 1, state[i] == [frame[i]]
            return np.array([current_frame])

        all_indices = np.arange(indices[0], self._end)
        last_game_terminal_index = all_indices[np.argmax(
            self.dones[all_indices % self.capacity]
        )]
        if self.dones[last_game_terminal_index] == True:
            validity = (
                indices >= max(self._end - self.size,  # in range
                               last_game_terminal_index + 1)
            )
        else:  # all records are non-terminal
            validity = (
                indices >= self._end - self.size  # in range
            )

        _frames = np.concatenate(
            (self.frames[indices[validity] % self.capacity],
             [current_frame]),
            axis=0
        )
        if len(_frames) < self.hist_len:
            _frames = np.concatenate(
                (np.zeros(shape=(self.hist_len - len(_frames), *self.frame_shape),
                          dtype=self.dtype),
                 _frames),
                axis=0
            )
        return _frames

    def sampleable_range(self, raise_for_empty_range=False):
        """
        Returns the indices that can represent a sample in current records.

        Note that the criterion for an index to be sampleable is that:
            _start - hist_index_shifts[0] <= index < _end - 1 (-1 for next frame),
        which does not ensure that the index is valid.

        :param raise_for_empty_range:
            If True, a RuntimeError will be raised when the sampleable range 
            is empty.

        """
        if not self.is_full and self._start <= self._end:
            start = self._start - self.hist_index_shifts[0]
            end = self._end - 1
        else:  # self._end < self._start:
            start = self._start - self.hist_index_shifts[0]
            end = self._end + self.capacity - 1

        if end <= start and raise_for_empty_range:
            # Actually this error will only be triggered when there are
            # `1 - self.hist_index_shifts[0]` records (i.e. when `self.add` is
            # called only once after the memory is initialized).
            # This is because `start < end` indicates that
            #     1 - self.hist_index_shifts[0] < self.size,
            # while `self.add`ensures that
            #     self.size >= 1 - self.hist_index_shifts[0]
            # and the equality only holds when self.add is called only once
            # after the memory is initialized.
            raise RuntimeError(
                f"Cannot sample a batch with {self.size} transitions "
                f"(size = {self.size}, capacity = {self.capacity}, "
                f"expected sample range = [{start}, {end}))."
            )

        return np.arange(start=start, stop=end, dtype=int) % self.capacity

    def sample(self, batch_size):
        """
        Samples a batch of (state, action, reward, done, next_state) transition.

        """
        sampled_indices = []
        sampleable_indices = self.sampleable_range()
        attempt_count = 0
        while (len(sampled_indices) < batch_size and
               attempt_count < self._max_sample_attempts):
            indices = np.random.choice(
                sampleable_indices,
                batch_size - len(sampled_indices)
            )
            for _index in indices:
                if self.is_valid_index(_index, allow_terminate_at_index=False):
                    sampled_indices.append(_index)
                else:
                    attempt_count += 1

        if len(sampled_indices) != batch_size:
            raise RuntimeError(
                f"Max sample attempts: Re-tried {attempt_count} times but only "
                f"{len(sampled_indices)} samples got (batch size = {batch_size})."
            )

        # shape=(batch_size, hist_len, *frame_shape)
        states = np.array(
            [self.get_state(index) for index in sampled_indices]
        )
        next_states = np.array(
            [self.get_state(index + 1) for index in sampled_indices]
        )

        return (
            states,
            self.actions[sampled_indices],
            self.rewards[sampled_indices],
            self.dones[sampled_indices],
            next_states
        )

    def info(self):
        return (
            f"size = {self.size} (capacity = {self.capacity}"
            f"{', full' if self.is_full else ''})\n" +
            f"cursors: ({self._start}, {self._end})"
        )


if __name__ == "__main__":
    transition = FrameTransition()
    memory = ReplayMemory(
        capacity=5, hist_len=3,
        hist_type="linear", hist_spacing=1
    )
