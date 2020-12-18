from itertools import product
import pickle
import time
from IPython import embed
import json
import argparse

import numpy as np
import torch

from memory import Memory
from qlearner import QLearning
from adp import CartPoleADP
from mutual import MutHook, HeterogeneousMutualLearner
from experiment import Experiment
import matplotlib.pyplot as plt

import gym

NBINS=9

eval_env = gym.make('CartPole-v1')

parser = argparse.ArgumentParser(
	description='Run an experiment'
)
parser.add_argument('fname')

def parse_config(fname):
	data = json.load(open(fname, 'r'))
	return data

if __name__ == '__main__':
	fname = parser.parse_args().fname
	cfg = parse_config(fname)
	exp = Experiment(cfg)

	exp.run()
	exp.save(fname)
