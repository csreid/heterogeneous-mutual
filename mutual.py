import torch
import numpy as np
from reinforcement import Learner
from adp import CartPoleADP
from qlearner import QLearning

class MutHook:
	def __init__(self, adp):
		self.adp = adp

	def __call__(self, s_s):
		av_s = [] #torch.zeros(len(s), 2)

		for idx, s in enumerate(s_s):
			av_s.append(self.adp.get_action_vals(s.detach().numpy()))

		with torch.no_grad():
			r = torch.tensor(np.array(av_s)).float()
		return r, self.adp._last_eval

class HeterogeneousMutualLearner(Learner):
	def __init__(
		self,
		primary='q',
		gamma=0.99,
		adp_delta=0.01,
		adp_bins=5,
		mutual_steps=1000,
		do_target_q=False,
		q_target_lag=100
	):
		self._mutual_steps = mutual_steps
		self._steps = 0
		self._adp = CartPoleADP(
			gamma=gamma,
			delta=adp_delta,
			nbins=adp_bins
		)
		self._q = QLearning(
			gamma=gamma,
			target=do_target_q,
			target_lag=q_target_lag
		)

		hook = MutHook(self._adp)
		self._q.set_mutual_hook(hook)
		self._q.set_mutual_steps(mutual_steps)

		if primary == 'q':
			self._primary = self._q
		elif primary =='adp':
			self._primary = self._adp
		else:
			raise Exception('Invalid option')

	def handle_transition(self, s, a, r, sp, done):
		self._steps += 1
		self._q.handle_transition(s, a, r, sp, done)
		if self._steps < self._mutual_steps:
			self._adp.handle_transition(s, a, r, sp, done)

	def get_action_vals(self, s):
		return self._primary.get_action_vals(s)

	def exploration_strategy(self, s):
		return self._primary.exploration_strategy(s)

	def deterministic_strategy(self, s):
		return self._primary.deterministic_strategy(s)

	def evaluate(self, env, n):
		adp_eval = self._adp.evaluate(env, n)
		q_eval = self._q.evaluate(env, n)

		if self._primary == self._q:
			return q_eval
		else:
			return adp_eval
