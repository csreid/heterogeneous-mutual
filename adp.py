import numpy as np
from numpy import e
from itertools import product

from memory import Memory
from reinforcement import Learner

import time

class CartPoleADP(Learner):
	mins = np.array([-4.8, -2, -0.418, -4])
	maxes = -mins

	def __init__(self, nbins=5, gamma=0.99, delta=0.01):
		super().__init__()
		# F maps state-action tuples to a count of the following states
		self.F = np.zeros(
			# Each state
			tuple([nbins+1 for _ in range(4)]) +
			# Each action
			(2,) +
			# Each Reward possibility
			(2,) +
			# Next states
			tuple([nbins+1 for _ in range(4)])
		)
		self.gamma = gamma
		self.bins = nbins
		self.delta = delta

		self.V = np.random.random((nbins+1, nbins+1, nbins+1, nbins+1))
		self.visits = np.zeros(self.V.shape)
		self._statemap = {}
		self._rewardmap = {}
		self._temp = 5000

		self.bounds = np.array([
			np.linspace(self.mins[i], self.maxes[i], nbins)
			for i in range(4)
		])

		for s in product(*[range(self.bins+1) for _ in range(4)]):
			for a in range(2):
				self._statemap[s + (a,)] = []
				self._rewardmap[s + (a,)] = []

	def _p(self, s, a, r, sp):
		# Probability of seeing state `sp` and getting reward `r` after taking action `a` in state `s`
		#s = self._convert_to_discrete(s)
		#sp = self._convert_to_discrete(sp)

		idx_sp = tuple(s) + (a,) + (r,) + tuple(sp)
		idx_s = tuple(s) + (a,)

		return (self.F[idx_sp]) / (np.sum(self.F[idx_s]))


	def do_pass(self):
		delta = 0
		for s in product(*[range(self.bins+1) for _ in range(4)]):
			if self.visits[s] == 0:
				continue
			v = self.V[s]
			self.V[s] = np.max(self.get_action_vals(s))
			delta = np.max([delta, np.abs(v - self.V[s])])

		return delta

	def get_action_vals(self, s):
		s = self._convert_to_discrete(s)

		#vals = np.array([
		#	np.sum([
		#		# Here, `r` will only be 0 in a terminal state or 1 otherwise
		#		# We take advantage of that to set the second term to 0 when `sp` is terminal
		#		self._p(s, a, r, sp) * (r + self.gamma * r * self.V[sp])
		#		for sp, r in product(set(self._statemap[s + (a,)]), range(2))
		#	])
		#	for a in range(2)
		#])
		vals = []
		for a in range(2):
			total=0
			for sp, r in product(set(self._statemap[tuple(s) + (a,)]), range(2)):
				total += self._p(s, a, r, sp) * (r + self.gamma * r * self.V[sp])

			vals.append(total)

		return np.array(vals)

	def exploration_strategy(self, s):
		s = self._convert_to_discrete(s)

		qs = self.get_action_vals(s)
		ps = [(e ** (q / self._temp)) / np.sum(e ** (qs / self._temp)) for q in qs]
		return np.random.choice(np.arange(len(ps)), p = ps)

	def deterministic_strategy(self, s):
		s = self._convert_to_discrete(s)

		qs = self.get_action_vals(s)

		return np.argmax(qs)

	def _convert_to_discrete(self, s):
		bounds = self.bounds
		return super()._convert_to_discrete(s, bounds)

	def handle_transition(self, s, a, r, sp, done):
		s = self._convert_to_discrete(s)
		sp = self._convert_to_discrete(sp)

		if done:
			r = 0

		idx = tuple(s) + (a,) + (int(r),) + tuple(sp)
		self.visits[tuple(s)] += 1
		self.F[idx] += 1

		if tuple(s) + (a,) in self._statemap:
			self._statemap[tuple(s) + (a,)].append(sp)
		else:
			self._statemap[tuple(s) + (a,)] = [sp]

		if tuple(s) + (a,) in self._rewardmap:
			self._rewardmap[tuple(s) + (a,)].append(r)
		else:
			self._rewardmap[tuple(s) + (a,)] = [r]

		self._temp = max(1, self._temp-1)

		while self.do_pass() > self.delta:
			pass
