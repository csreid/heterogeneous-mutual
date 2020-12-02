import time
import copy

import numpy as np
from numpy import e
import torch
from torch.nn import Sequential, Linear, LeakyReLU, MSELoss
from torch.optim import Adam

from memory import Memory
from reinforcement import Learner
from scipy.special import softmax

np.seterr(all='raise')

class QLearning(Learner):
	def __init__(
		self,
		opt=Adam,
		loss=MSELoss,
		gamma=0.99,
		mutual_steps=1000,
		lr=0.01,
		target=False,
		target_lag=100
	):
		self._memory = Memory(1000, (4,))
		self.Q = Sequential(
			Linear(4, 8),
			LeakyReLU(),
			Linear(8, 8),
			LeakyReLU(),
			Linear(8,2)
		)

		if target:
			self._target = True
			self.target_Q = copy.deepcopy(self.Q)
			self._lag = target_lag
		else:
			self._target = False

		self.gamma = gamma

		self.opt = opt(self.Q.parameters(), lr=lr)
		self.mut_opt = opt(self.Q.parameters(), lr=lr)
		self._base_opt = opt
		self._lr = lr
		self._base_loss_fn = MSELoss()
		self._steps = 0

		self._temp = 5000
		self._mutual_hook = None
		self._mutual_agents = []
		self._mutual_steps = mutual_steps

		self._do_mutual = True

		self.eps = 1.0
		self.decay = 0.9999
		self._loss_history = []

	def set_mutual_steps(self, steps):
		self._mutual_steps = steps

	def _mutual_loss_weight(self):
		return (-1 / (1 + np.e ** ((-(self._steps - self._mutual_steps))/500))) + 1
		#return softmax(np.array([self._mutual_hook.adp._last_eval, self._last_eval]))

	def compute_mutual_loss(self, y_pred, X):
		if self._mutual_hook is None:
			return 0

		mut, evl = self._mutual_hook(X)

		if evl is None or self._last_eval is None:
			return 5 * self._base_loss_fn(mut, y_pred)# * self._mutual_loss_weight()

		return (evl / self._last_eval) * self._base_loss_fn(mut, y_pred)# * self._mutual_loss_weight()

	def learn(self, batch_size=32, n_samples=32):
		if len(self._memory) < n_samples:
			return 'n/a'

		X, y = self._build_dataset(n_samples)
		y_pred = self.Q(X)
		td_loss = self._base_loss_fn(y, y_pred)

		if self._steps < self._mutual_steps:
			mut_loss = self.compute_mutual_loss(y_pred, X)
		else:
			mut_loss = 0

		loss = td_loss + mut_loss
		self._loss_history.append((td_loss, mut_loss, loss))

		self.opt.zero_grad()
		loss.backward()
		self.opt.step()

		return loss.item()

	def _build_dataset(self, n):
		with torch.no_grad():
			s_s, a_s, r_s, sp_s, done_mask = self._memory.sample(n)

			if self._target:
				Q = self.target_Q
			else:
				Q = self.Q

			vhat_sp_s = torch.max(Q(sp_s.float()), dim=1).values
			vhat_sp_s[done_mask] = 0

			targets = self.Q(s_s.float())

			for idx, t in enumerate(targets):
				t[int(a_s[idx].byte())] = r_s[idx] + self.gamma * vhat_sp_s[idx]

			X = s_s.float()
			y = targets
		return X, y

	def handle_transition(self, s, a, r, sp, done):
		s = self._convert_to_torch(s)
		sp = self._convert_to_torch(sp)

		self._memory.append((
			s,
			torch.from_numpy(np.array([a]))[0],
			r,
			sp,
			done
		))
		self.learn()
		self._steps += 1

		if self._target and (self._steps % self._lag) == 0:
			self.target_Q = copy.deepcopy(self.Q)

		for agt in self._mutual_agents:
			agt.handle_transition(s, a, r, sp, done)

	def get_action_vals(self, s):
		s = self._convert_to_torch(s)

		return self.Q(s)

	def exploration_strategy(self, s):
		self.eps *= self.decay
		if np.random.random() > self.eps:
			best_action = self.deterministic_strategy(s)
			return best_action
		else:
			ps = np.full(2, 0.5)
			return np.random.choice(np.arange(len(ps)), p=ps)

		return ps

	def set_mutual_hook(self, hook):
		"""
			A function to set a hook into another learning system; `hook` should be a callable that
			takes a set of state values `s` and returns their corresponding `V(s)`, along with
			the latest evaluation score for the mutual agent
		"""
		self._mutual_hook = hook

	def set_mutual_agents(self, agts):
		"""
			A `list` of other agents on which this agent should call `handle_transition` when it
			itself calls `handle_transition`
		"""
		self._mutual_agents = agts

	def deterministic_strategy(self, s):
		s = self._convert_to_torch(s)

		vals = self.get_action_vals(s)
		return torch.argmax(vals).item()
