import torch
import numpy as np

class Memory:
	def __init__(self, max_len, shape):
		self.s_s = torch.zeros((max_len,) + shape, requires_grad=False)
		self.a_s = torch.zeros((max_len), requires_grad=False)
		self.sp_s = torch.zeros((max_len,) + shape, requires_grad=False)
		self.r_s = torch.zeros((max_len), requires_grad=False)
		self.done_mask = np.zeros(max_len, dtype=bool)
		self._counter = 0
		self.max_len = max_len

	def __len__(self):
		return min(self._counter, self.max_len)

	def __iter__(self):
		self._n = 0

		return self

	def __next__(self):
		if self._n >= min(self._counter, self.max_len):
			raise StopIteration

		val = (
			self.s_s[self._n],
			self.a_s[self._n],
			self.r_s[self._n],
			self.sp_s[self._n],
			self.done_mask[self._n]
		)
		self._n += 1

		return val

	def append(self, t):
		i = self._counter
		s, a, r, sp, done = t
		idx = i % self.max_len

		self.s_s[idx] = s
		self.a_s[idx] = a
		self.r_s[idx] = r
		self.sp_s[idx] = sp
		self.done_mask[idx] = done

		self._counter += 1

	def sample(self, n):
		idx = np.random.randint(0, min(self._counter, self.max_len), size=n)

		return (
			self.s_s[idx],
			self.a_s[idx],
			self.r_s[idx],
			self.sp_s[idx],
			self.done_mask[idx]
		)
