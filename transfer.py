import torch
from torch.optim import Adam
from torch.nn import Sequential, Linear, LeakyReLU, MSELoss
import numpy as np
import pickle

from adp import CartPoleADP
from qlearner import QLearning
from memory import Memory
import gym

Q = Sequential(
	Linear(4, 8),
	LeakyReLU(),
	Linear(8, 8),
	LeakyReLU(),
	Linear(8, 2)
)
opt = Adam(Q.parameters())
loss_fn = MSELoss()
memory = Memory(1000, (4,))

adp_agt = CartPoleADP(nbins=9, gamma=0.99, delta=0.01)

def get_avs(X):
	av_s = []
	for idx, s in enumerate(X):
		av_s.append(adp_agt.get_action_vals(s.detach().numpy()))

	return torch.tensor(np.array(av_s)).float()

env = gym.make('CartPole-v1')
eval_env = gym.make('CartPole-v1')

done = False
s = env.reset()
for step in range(1010):
	a = adp_agt.get_action(s)
	sp, r, done, _ = env.step(a)

	adp_agt.handle_transition(s, a, r, sp, done)
	memory.append((
		torch.tensor(s),
		a,
		r,
		torch.tensor(sp),
		done
	))
	s = sp

	if done:
		print(f'Done with {step} steps')
		s = env.reset()
		done = False

print(f'ADP eval: {adp_agt.evaluate(eval_env, 10)}')
q_agt = QLearning()
q_agt.Q = Q
q_agt.opt = opt
q_agt._memory = memory

for i in range(100):
	losses = []
	for (s, a, r, sp, done) in memory:
		y = torch.tensor(adp_agt.get_action_vals(s.detach().numpy())).float()
		y_pred = Q(s)

		loss = loss_fn(y, y_pred)

		opt.zero_grad()
		loss.backward()
		opt.step()

		losses.append(loss.detach())

	print(np.mean(losses))

print(f'Q eval: {q_agt.evaluate(eval_env, 10)}')
