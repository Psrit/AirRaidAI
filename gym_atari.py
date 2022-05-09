import sys
import typing

import gym

if sys.version_info >= (3, 8):
    @typing.runtime_checkable
    class EnvProtocol(typing.Protocol):
        """
        The protocol of the game environment.

        """

        def render(self):
            ...

        def reset(self):
            ...

        def step(self, action) -> typing.Tuple[
            object, float, bool, dict
        ]:
            """
            Move one step in the game environment, and return
            (observation, reward, done, info)

            The shape of the observation space of `env` must be a 3-tuple, e.g.
            (width of the image, height of the image, number of channels).

            """
            ...

else:
    EnvProtocol = typing.TypeVar("EnvProtocol", bound=gym.Env)
