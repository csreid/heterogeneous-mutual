import time
from rebar.learners.qlearner import QLearner
from rebar.learners.adp import ADP
import numpy as np
import gym
import torch
from envs import Swingup, Reacher, InvertedDoublePendulum, InvertedPendulum
from copy import deepcopy
from gym.wrappers import TimeLimit

env = Reacher

agt = ADP(
	action_space = env.action_space,
	observation_space = env.observation_space,
	bins=5000,
	initial_temp=10000,
	delta=0.01
)

s = env.reset()
disc_state = agt._convert_to_discrete(s)
cont_state = agt.sample_state(s, 1)[0]
