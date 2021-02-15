import torch
import numpy as np
from rebar.learners.learner import Learner
from torch.nn.functional import softmax, cosine_similarity
from torch.nn import MSELoss, KLDivLoss

from rebar.learners.qlearner import QLearner
from rebar.learners.adp import ADP

class MutHook:
	def __init__(self, adp):
		self.adp = adp

	def __call__(self, s_s):
		av_s = []

		for idx, s in enumerate(s_s):
			av_s.append(self.adp.get_action_vals(s.detach().numpy()))

		with torch.no_grad():
			r = torch.tensor(np.array(av_s)).float()
		return r, self.adp._last_eval

class HeterogeneousMutualLearner(Learner):
	def __init__(
		self,
		action_space,
		observation_space,
		primary='q',
		gamma=0.99,
		adp_delta=0.01,
		adp_bins=7,
		mutual_steps=1000,
		do_target_q=False,
		q_target_lag=100,
		model_lag=100,
		initial_epsilon=1.0,
		final_epsilon=0.01,
		epsilon_decay_steps=5000
	):
		self._mutual_steps = mutual_steps
		self._mutual_loss_fn = KLDivLoss(reduction='batchmean')
		self._steps = 0
		self._adp = ADP(
			action_space=action_space,
			observation_space=observation_space,
			bins=adp_bins,
			gamma=gamma,
			delta=adp_delta
		)
		self._q = QLearner(
			action_space=action_space,
			observation_space=observation_space,
			Q='simple',
			opt_args={
				'lr': 0.01
			},
			gamma=gamma,
			memory_len=1000,
			target_lag=q_target_lag,
			initial_epsilon=initial_epsilon,
			final_epsilon=final_epsilon,
			exploration_steps=epsilon_decay_steps
		)

		self.model_lag = model_lag

		if primary == 'q':
			self._primary = self._q
		elif primary =='adp':
			self._primary = self._adp
		else:
			raise Exception('Invalid option')

	def handle_transition(self, s, a, r, sp, done):
		self._steps += 1
		if self._steps < self._mutual_steps:
			self._handle_mutual(s)

		self._q.handle_transition(s, a, r, sp, done)
		self._adp.handle_transition(s, a, r, sp, done)

	def _handle_mutual(self, s):
		q_greedy = self._q.exploitation_policy(s)
		adp_greedy = self._adp.exploitation_policy(s)
		adp_confidence = self._adp.confidence(s)

		if q_greedy == adp_greedy:
			return

		data = self._adp.sample_state(s, 64)
		y = []
		for d in data:
			y.append(self._adp.get_action_vals(d))
		y = softmax(torch.tensor(y).float(), dim=1)

		y_pred = softmax(self._q.Q(s), dim=1)

		l = self._mutual_loss_fn(y, y_pred.repeat(64, 1))
		l_weight = torch.sum(cosine_similarity(torch.tensor(data), s.repeat(64, 1), dim=1))

		loss = l * torch.sqrt(l_weight * adp_confidence)

		self._q.opt.zero_grad()
		loss.backward()
		self._q.opt.step()

	def get_action_vals(self, s):
		return self._primary.get_action_vals(s)

	def exploration_policy(self, s):
		return self._primary.exploration_policy(s)

	def exploitation_policy(self, s):
		return self._primary.exploitation_policy(s)

	def evaluate(self, env, n):
		adp_eval = self._adp.evaluate(env, n)
		q_eval = self._q.evaluate(env, n)

		if self._primary == self._q:
			return q_eval
		else:
			return adp_eval
