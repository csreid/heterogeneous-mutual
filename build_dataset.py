import time
from rebar.learners.qlearner import QLearner
from rebar.learners.adp import ADP
from rebar.memory import Memory
import numpy as np
import gym
import torch
import pickle
from envs import Swingup, Reacher, InvertedDoublePendulum, InvertedPendulum, Walker
from copy import deepcopy
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt

env = InvertedPendulum
play_env = TimeLimit(deepcopy(env), max_episode_steps=2000)
play_env.render()

agt = ADP(
	action_space=env.action_space,
	observation_space=env.observation_space,
	bins=5,
	gamma=0.99,
	initial_temp=2000,
	delta=0.01
)

def play(agt, env):
	done = False
	s = env.reset()
	total_r = 0
	while not done:
		a = agt.get_action(s, explore=False)
		s, r, done, _ = env.step(a)
		total_r += r
		time.sleep(1./30.)

	return total_r

s = env.reset()
m = Memory(
	max_len=1000,
	obs_shape=(5,),
	action_shape=(1,)
)

for step in range(1000):
	a = agt.get_action(s)
	sp, r, done, _ = env.step(a)

	agt.handle_transition(s, a, r, sp, done)
	m.append((s, a, r, sp, done))

	if done:
		s = env.reset()
		done = False

	s = sp

pickle.dump(m, open('dataset.pkl', 'wb'))
print(play(agt, play_env))
