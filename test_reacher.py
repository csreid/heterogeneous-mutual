import time
from rebar.learners.qlearner import QLearner
from rebar.learners.adp import ADP
import numpy as np
import gym
import torch
from envs import Swingup, Reacher, InvertedDoublePendulum, InvertedPendulum, Walker
from copy import deepcopy
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt

env = Reacher
eval_env = TimeLimit(deepcopy(env), max_episode_steps=500)
play_env = TimeLimit(deepcopy(env), max_episode_steps=200)
env = TimeLimit(deepcopy(env), max_episode_steps=1000)
play_env.render()

# Swingup state = <x, vx, cos(theta), sin(theta), thetadot>

q = QLearner(
	action_space = env.action_space,
	observation_space = env.observation_space,
	Q = 'simple',
	opt_args={ 'lr': 0.01 },
	memory_len=1000,
	gamma=0.999,
	initial_epsilon=1.,
	final_epsilon=0.01,
	exploration_steps=50000,
	target_lag=100
)

adp = ADP(
	action_space = env.action_space,
	observation_space = env.observation_space,
	bins=13,
	gamma=0.99,
	initial_temp=50000,
	delta=0.01
)

agt = adp

s_s = []

def play(agt, env):
	done = False
	s = env.reset()
	s_s = [agt._convert_to_discrete(s)]
	while not done:
		a = agt.get_action(s, explore=False)
		s, r, done, _ = env.step(a)
		s_s.append(np.array(agt._convert_to_discrete(s)))
		time.sleep(1./30.)

	state_vals = ['cos', 'sin', 'x-pos', 'x-vel', 'y-pos', 'y-vel', 'to-target']
	for line, label in zip(np.array(s_s).T, state_vals):
		plt.plot(line, label=label)

	plt.legend()
	plt.show()

	return np.array(s_s).T

#s_s = play(agt, play_env)
#labels = ['x', 'v_x', 'cos(theta)', 'sin(theta)', 'thetadot']
#for label, line in zip(labels, s_s):
#	plt.plot(line, label=label)
#plt.legend()
#plt.show()

s = env.reset()
for step in range(100000):
	a = int(q.get_action(s))
	sp, r, done, _ = env.step(a)

	agt.handle_transition(s, a, r, sp, done)

	s_s.append(s.detach().numpy())

	s = sp

	if done:
		s = env.reset()
		done = False

	if (step % 1000) == 0:
		print(f'{step}: {adp.evaluate(eval_env, 10)} (adp) {q.evaluate(eval_env, 10)} (Q)')
		play(agt, play_env)
