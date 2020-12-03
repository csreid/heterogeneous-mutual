import numpy as np
import torch
import time

class Learner:
	def __init__(self):
		self._temp = 5000
		self._last_eval = None
		self.name = None
		self._evals = []

	def get_action(self, s, explore=True):
		if explore:
			return self.exploration_strategy(s)

		return self.deterministic_strategy(s)

	def evaluate(self, env, n, starting_state=None):
		vals = []
		for _ in range(n):
			done = False
			s = env.reset()

			if starting_state is not None:
				s = starting_state
				env.env.state = s

			total_r = 0
			steps = 0

			while not done:
				a = self.get_action(s, explore=False)
				s, r, done, _ = env.step(a)
				total_r += r
				steps += 1

				if (steps > 500):
					done = True
			vals.append(total_r)

		evl = np.mean(np.array(vals))
		self._last_eval = evl

		self._evals.append(evl)
		return np.mean(np.array(vals))

	def play(self, env):
		done = False
		s = env.reset()
		total_r = 0

		while not done:
			a = self.get_action(s, explore=False)
			print(self.get_action_vals(s))
			s, r, done, _ = env.step(a)
			total_r += r

			env.render()
			time.sleep(1./60.)

		env.close()

	def _convert_to_discrete(self, s, bounds):
		if type(s) is tuple:
			return s

		if torch.is_tensor(s):
			s = s.detach()

		new_obs = tuple(
			np.searchsorted(self.bounds[i], s[i])
			for i in range(4)
		)

		return new_obs

	def _convert_to_torch(self, s):
		if torch.is_tensor(s):
			return s

		new_s = torch.tensor(s, requires_grad=True).float().reshape((-1))
		return new_s

	def set_name(self, name):
		self.name = name

	def handle_transition(self, s, a, r, sp, done):
		raise NotImplementedError

	def get_action_vals(self, s):
		raise NotImplementedError

	def exploration_strategy(self, s):
		raise NotImplementedError

	def deterministic_strategy(self, s):
		raise NotImplementedError
