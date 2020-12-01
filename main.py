from itertools import product
import pickle
import time

import numpy as np
import torch

from memory import Memory
from qlearner import QLearning
from adp import CartPoleADP
from mutual import MutHook, HeterogeneousMutualLearner
import matplotlib.pyplot as plt

#torch.autograd.set_detect_anomaly(True)

import gym

NBINS=9

eval_env = gym.make('CartPole-v1')

def do_step(agt, env, s):
	a = agt.get_action(s)
	sp, r, done, _ = env.step(a)

	agt.handle_transition(s, a, r, sp, done)

	if done:
		done = False
		sp = env.reset()

	return sp

def grab_states(env, n):
	done = False
	s = env.reset()
	states = torch.zeros((n,) + env.observation_space.shape)

	for idx in range(n):
		states[idx] = torch.tensor(s)
		s, r, done, _ = env.step(env.action_space.sample())

		if done:
			s = env.reset()
			done = False

	return states

def do_experiment(
	fname,
	ml_type=None,
	do_baseline=False,
	do_mutual=False,
	n_trials=50,
	n_steps=5000,
	mutual_steps=1000,
	delta=0.01,
	steps_per_eval=200
):
	trial_rs = []
	all_mean_held_states = []
	states = grab_states(gym.make('CartPole-v1'), 100)

	for trial in range(n_trials):
		agts = []
		rs = []
		mean_held_states = []

		adp = CartPoleADP(nbins=NBINS, gamma=0.9)
		qlrn = QLearning(
			#target=True,
			#target_lag=100
		)

		if ml_type == 'share_tuples':
			qlrn.set_mutual_agents([adp])
			adp.set_name('adp_sharing_tuples')

		if ml_type == 'share_qs':
			hook = MutHook(adp)
			qlrn.set_mutual_hook(hook)
			qlrn.set_name('q_mutual')
			qlrn.set_mutual_steps(mutual_steps)

		if ml_type == 'both':
			qlrn.set_mutual_agents([adp])
			hook = MutHook(adp)
			qlrn.set_mutual_hook(hook)
			qlrn.set_name('q_mutual')
			qlrn.set_mutual_steps(mutual_steps)
			adp.set_name('adp_sharing_tuples')

		if ml_type is not None:
			agts.append(qlrn)
			agts.append(adp)

		if do_baseline == 'both':
			qlrn_normal = QLearning(
				#target=True,
				#target_lag=100
			)
			adp_normal = CartPoleADP(nbins=NBINS, gamma=0.9)
			qlrn_normal.set_name('q_standard')
			adp_normal.set_name('adp_standard')
			agts.append(qlrn_normal)
			agts.append(adp_normal)

		if do_baseline == 'q':
			qlrn_normal = QLearning(
				#target=True,
				#target_lag=100
			)
			qlrn_normal.set_name('q_standard')
			agts.append(qlrn_normal)

		if do_baseline == 'adp':
			qlrn_normal = CartPoleADP(nbins=NBINS, gamma=0.9)
			qlrn_normal.set_name('adp_standard')
			agts.append(qlrn_normal)

		if do_mutual:
			mutual = HeterogeneousMutualLearner(
				mutual_steps=mutual_steps,
				do_target_q = True,
				q_target_lag=100,
				adp_bins=NBINS
			)
			mutual.set_name('single_agent_mutual')
			agts.append(mutual)

		envs = [gym.make('CartPole-v1').env for _ in agts]
		s_s = [env.reset() for env in envs]

		for step in range(n_steps):
			for idx, (agt, env, s) in enumerate(zip(agts, envs, s_s)):
				s_s[idx] = do_step(agt, env, s)

			if (step % steps_per_eval) == 0:
				evls = [agt.evaluate(eval_env, 20) for agt in agts]
				rs.append(evls)

			if (step % 10) == 0:
				mean_held_states.append([torch.mean(agt.Q(states)) for agt in agts if agt.name in ['q_mutual', 'q_standard']])
				evl_str = ' | '.join([f'{a.name}: {e}' for (a, e) in zip(agts, evls)])
				print(f'Trial {trial}\t\tStep {step}  \t\tEvals {evl_str}')

		trial_rs.append(rs)
		all_mean_held_states.append(mean_held_states)

	fname = f'./results/{time.time()}_{fname}.pickle'
	pickle.dump({
		'names': [agt.name for agt in agts],
		'results': trial_rs,
		'held_states': all_mean_held_states
	}, open(fname, 'wb'))

if __name__ == '__main__':
	states = grab_states(gym.make('CartPole-v1'), 100)
	do_experiment(
		fname='foo',
		n_steps=5000,
		mutual_steps=1500,
		n_trials=5,
		do_baseline='q',
		ml_type=None,
		do_mutual=True,
		steps_per_eval=200
	)
