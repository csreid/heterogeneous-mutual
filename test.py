import torch
import numpy as np
from IPython import embed
from mutual import MutHook, HeterogeneousMutualLearner, QLearning
from randomagent import RandomAgent
import matplotlib.pyplot as plt
import gym

env1 = gym.make('CartPole-v1').env
env2 = gym.make('CartPole-v1').env
env3 = gym.make('CartPole-v1').env
eval_env = gym.make('CartPole-v1')
agt = HeterogeneousMutualLearner(
	mutual_steps=1500,
	do_target_q = True,
	q_target_lag=100,
	adp_bins=9,
	adp_gamma=0.99
)

std_agt = QLearning()
target_agt = QLearning(
	target=True,
	target_lag=100
)

def do_step(agt, env, s):
	a = agt.get_action(s)
	sp, r, done, _ = env.step(a)

	agt.handle_transition(s, a, r, sp, done)

	if done:
		done = False
		sp = env.reset()

	return sp

# Grab random some states
rand = RandomAgent(eval_env.observation_space, eval_env.action_space)
s = eval_env.reset()
done = False
for step in range(1000):
	s = do_step(rand, eval_env, s)
X = rand.memory.sample(100)[0]

std_est_vals = []
mut_est_vals = []
target_est_vals = []

evals_std = []
evals_mut = []
evals_target = []

def run(n):
	done = False
	s1 = env1.reset()
	s2 = env2.reset()
	s3 = env3.reset()
	for step in range(n):
		print(step)
		s1 = do_step(agt, env1, s1)
		s2 = do_step(std_agt, env2, s2)
		s3 = do_step(target_agt, env3, s3)

		if (step % 100) == 0:
			evl = agt.evaluate(eval_env, 10)
			evals_mut.append(evl)
			mut_est_vals.append(torch.mean(agt._q.Q(X)))

			evl = std_agt.evaluate(eval_env, 10)
			evals_std.append(evl)
			std_est_vals.append(torch.mean(std_agt.Q(X)))

			evl = target_agt.evaluate(eval_env, 10)
			evals_target.append(evl)
			target_est_vals.append(torch.mean(target_agt.Q(X)))

def do_plot():
	fig, (score_ax,est_ax,target_ax) = plt.subplots(nrows=3)

	score_ax.plot(evals_mut, label='Mutual')
	score_ax.plot(evals_std, label='Standard')
	score_ax.plot(evals_target, label='Target')
	score_ax.set_ylabel('Evaluation Score')
	score_ax.legend()

	est_ax.set_ylabel('Mean State Estimate')
	est_ax.plot(std_est_vals, label='Standard')
	est_ax.plot(mut_est_vals, label='Mutual')
	est_ax.plot(target_est_vals, label='Target')
	est_ax.legend()

	score_ax.axvline(1500 / 100)
	est_ax.axvline(1500 / 100)

	plt.show()

run(5000)
do_plot()
embed()
