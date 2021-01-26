import torch
from torch.nn import MSELoss, KLDivLoss
from rebar.learners.qlearner import QLearner
from torch.nn.functional import softmax
import gym

env = gym.make('CartPole-v1')

agt = QLearner(
	action_space=env.action_space,
	observation_space=env.observation_space,
	Q='simple',
	gamma=0.99,
	target_lag=100
)
loss_fn = KLDivLoss(reduction='batchmean')

s = torch.tensor(env.reset()).float()
adp_estimate = torch.tensor([62.0, 3.2241])

initial_sm = softmax(agt.Q(s), dim=1)[0]
target = softmax(adp_estimate, dim=0)

print(f'Initial Softmax: {initial_sm}')
print(f'Initial values: {agt.Q(s)}')
print(f'Target: {softmax(adp_estimate, dim=0)}')

print('Fitting...')
for _ in range(1000):
	sm = softmax(agt.Q(s), dim=1)[0]
	loss = loss_fn(sm, target)

	agt.opt.zero_grad()
	loss.backward()
	agt.opt.step()

result_sm = softmax(agt.Q(s), dim=1)[0]
print(f'Initial: {initial_sm}')
print(f'Result: {result_sm}')
print(f'Resulting values: {agt.Q(s)}')
print(f'Target: {softmax(adp_estimate, dim=0)}')
