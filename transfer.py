import torch
from torch.optim import Adam
from torch.nn import Sequential, Linear, LeakyReLU, MSELoss
import numpy as np
import pickle

from adp import CartPoleADP
from qlearner import QLearning
from memory import Memory
import gym

loss_fn = MSELoss()
memory = Memory(1000, (4,))

q_agt = QLearning()
Q = q_agt.Q
opt = q_agt.opt
q_agt.opt = opt
q_agt._memory = memory

adp_agt = CartPoleADP(nbins=9, gamma=0.99, delta=0.01)

def get_avs(X):
	av_s = []
	for idx, s in enumerate(X):
		av_s.append(adp_agt.get_action_vals(s.detach().numpy()))

	return torch.tensor(np.array(av_s)).float()

env = gym.make('CartPole-v1')
eval_env = gym.make('CartPole-v1')

held_states = torch.zeros(20, 4)
for i in range(20):
	s = torch.tensor(eval_env.reset())
	held_states[i] = s

done = False
s = env.reset()
for step in range(500):
	a = adp_agt.get_action(s)
	sp, r, done, _ = env.step(a)

	adp_agt.handle_transition(s, a, r, sp, done)
	q_agt.handle_transition(s, a, r, sp, done)
	s = sp

	if done:
		print(f'Done with {step} steps')
		s = env.reset()
		done = False

def print_info():
	print('--===============--')
	print(f'ADP eval: {adp_agt.evaluate(eval_env, 500)}')
	print(f'Q eval: {q_agt.evaluate(eval_env, 500)}')
	q_vals = Q(held_states)
	q_vals = torch.mean(torch.max(q_vals, dim=1).values)
	adp_vals = []
	for state in held_states:
		idx = tuple(adp_agt._convert_to_discrete(state.detach().numpy()))
		val = adp_agt.V[idx]
		adp_vals.append(val)
	print(f'Q vals: {q_vals}')
	print(f'ADP vals: {np.mean(adp_vals)}')
	print('--===============--')

print_info()

loss_delta = -float('Inf')
prev_loss = None
epochs = 0
while (prev_loss is None) or loss_delta < 0:
	losses = []
	for (s, a, r, sp, done) in memory:
		s = s.detach()
		y = torch.tensor(adp_agt.get_action_vals(s.detach().numpy())).float().detach()
		y_pred = Q(s)

		loss = loss_fn(y, y_pred)

		opt.zero_grad()
		loss.backward()
		opt.step()

		losses.append(loss.detach())

	epochs += 1

	loss = np.mean(losses)
	if prev_loss is not None:
		loss_delta = loss - prev_loss

	if (i % 100) == 0:
		print(f'{i}: {np.mean(losses)}')

	prev_loss = loss
print_info()
