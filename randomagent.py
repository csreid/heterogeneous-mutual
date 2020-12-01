from reinforcement import Learner
from memory import Memory

class RandomAgent(Learner):
	def __init__(
		self,
		observation_space,
		action_space,
		memory_len=1000
	):
		self.action_space = action_space
		self.memory = Memory(memory_len, observation_space.shape)

	def handle_transition(self, s, a, r, sp, done):
		s = self._convert_to_torch(s)
		sp = self._convert_to_torch(sp)

		self.memory.append((s, a, r, sp, done))
		pass

	def exploration_strategy(self, s):
		return self.action_space.sample()

	def deterministic_strategy(self, s):
		return self.action_space.sample()
