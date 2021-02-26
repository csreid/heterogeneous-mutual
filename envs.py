import gym
from gym import ObservationWrapper, ActionWrapper
from gym.spaces import Discrete, Box
import pybulletgym
import torch

import numpy as np

class TorchWrapper(ObservationWrapper):
	def observation(self, obs):
		return torch.tensor(obs).float()

class DiscreteActions(ActionWrapper):
	def __init__(self, env, bins):
		super().__init__(env)
		self._n_action_bins = bins
		mins = self.action_space.low
		maxes = self.action_space.high

		self.mins = mins
		self.maxes = maxes

		n_actions = bins ** self.action_space.shape[0]
		binwidth = (mins - maxes) / bins

		self.bins = np.array([
			np.linspace(mins[j], maxes[j], num=bins)
			for j
			in range(self.action_space.shape[0])
		])

		self.action_space = Discrete(n_actions)

	def action(self, a):
		i = 0
		tmp = a
		new_action = np.copy(self.mins)

		while tmp > 0:
			action = self.bins[i][tmp % self._n_action_bins]
			new_action[i] = action
			i += 1
			tmp = tmp // self._n_action_bins

		return tuple(new_action)

InvertedPendulum = DiscreteActions(TorchWrapper(gym.make('InvertedPendulumPyBulletEnv-v0').env), bins=3)
InvertedPendulum.observation_space.low = np.array([-1.001849, -1.5067716, 0.9800714, -0.19838282, -2.5435898])
InvertedPendulum.observation_space.high = np.array([1.0109291, 1.7392993, 1., 0.19864552, 3.2698603])

Swingup = DiscreteActions(TorchWrapper(gym.make('InvertedPendulumSwingupPyBulletEnv-v0').env), bins=3)
Swingup.observation_space.low = np.array([-0.8, -4.734085, -1., -0.9999993, -5.])
Swingup.observation_space.high = np.array([0.8,  5.269364,  1.,  0.99999994, 5.])

InvertedDoublePendulum = DiscreteActions(TorchWrapper(gym.make('InvertedDoublePendulumPyBulletEnv-v0').env), bins=3)
InvertedDoublePendulum.observation_space.low = np.array([-1.0825137e+00, -6.2160158e+00, -1.5882798e+00, 5.8954906e-01, -8.0773258e-01, -7.4234476e+00, 5.0194468e-03, -9.9998742e-01, -1.1855822e+01])
InvertedDoublePendulum.observation_space.high = np.array([1.089619, 6.4763017, 1.5819458, 1., 0.7905721, 7.691252, 1., 0.99988455, 11.297818])

Reacher = DiscreteActions(TorchWrapper(gym.make('ReacherPyBulletEnv-v0').env), bins=3)
Reacher.observation_space.low = np.array([0, -0.03903592, -0.25239912, -0.17094594, -1., -1., -2.4577107, -1.1436158, -10.])
Reacher.observation_space.high = np.array([0.04250292, -0.03903592, 0.16746053, 0.2490324, 1., 1., 10., 1.133179, 10.])

Walker = DiscreteActions(TorchWrapper(gym.make('Walker2DPyBulletEnv-v0').env), bins=3)
