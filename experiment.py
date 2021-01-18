import gym
from qlearner import QLearning
from adp import CartPoleADP
from mutual import MutHook, HeterogeneousMutualLearner
import torch
import pickle
from multiprocessing import Pool
import time
import numpy as np

class Experiment:
	def __init__(self, cfg):
		self.cfg = cfg

		self.use_target_q = cfg['useTargetQ']
		self.q_target_lag = cfg['QTargetLag']
		self.do_mutual = cfg['mutual']['ADPQ']
		self.do_hetmut = cfg['mutual']['heterogeneous']
		self.mutual_steps = cfg['mutual']['mutualSteps']
		self.mutual_weight = cfg['mutual']['weight']
		self.mutual_type = cfg['mutual']['type']

		self.initial_eps = cfg['exploration']['initialEpsilon']
		self.final_eps = cfg['exploration']['finalEpsilon']
		self.decay_steps = cfg['exploration']['decaySteps']

		self.do_standard_q = cfg['standard']['Q']
		self.do_standard_adp = cfg['standard']['ADP']

		self.n_trials = cfg['trials']
		self.steps = cfg['stepsPerTrial']
		self.steps_per_eval = cfg['stepsPerEvaluation']
		self.gamma = cfg['gamma']
		self.adp_bins = cfg['adpBins']

		self.env_name = cfg['envName']
		if type(cfg['holdOutStates']) == bool and cfg['holdOutStates']:
			self.hold_out_states = 100
		elif type(cfg['holdOutStates']) == bool:
			self.hold_out_states = None
		else:
			self.hold_out_states = cfg['holdOutStates']

		if type(self.hold_out_states) == int:
			self.held_states = self._get_held_states()
			self.val_history = np.zeros((
				self._n_agents(),
				self.n_trials,
				int(np.floor(self.steps / self.steps_per_eval))
			))

		self.results = []
		self.names = None

		self.step = 0

	def _n_agents(self):
		return len(self.get_names())

	def _do_step(self, agt, env, s):
		a = agt.get_action(s)
		sp, r, done, _ = env.step(a)
		agt.handle_transition(s, a, r, sp, done)

		if done:
			sp = env.reset()

		self.step += 1
		return sp

	def save(self, fname=''):
		if len(fname) > 0:
			fname = '_' + fname

		fname = f'./results/{time.time()}{fname}.pickle'
		pickle.dump({
			'names': self.get_names(),
			'results': self.results,
			'config': self.cfg,
			'val_history': self.val_history
		}, open(fname, 'wb'))

	def _get_held_states(self):
		n = self.hold_out_states
		env = gym.make(self.env_name)
		s_s = torch.zeros((n,) + env.observation_space.shape)
		for idx in range(n):
			s = torch.tensor(env.reset())
			s_s[idx] = s

		return s_s

	def get_names(self):
		names = []

		if self.do_mutual:
			names.append('adp_sharing_tuples')
			names.append('q_mutual')

		if self.do_standard_q:
			names.append('q_standard')

		if self.do_standard_adp:
			names.append('adp_standard')

		if self.do_hetmut:
			names.append('single_agent_mutual')

		return names

	def _do_trial(self, i):
		agts = self._build_agts()
		envs = [gym.make(self.env_name).env for _ in agts]
		eval_envs = [gym.make(self.env_name) for _ in agts]
		s_s = [env.reset() for env in envs]
		trial_evals = []

		for step in range(self.steps):
			for idx, (agt, env, s) in enumerate(zip(agts, envs, s_s)):
				s_s[idx] = self._do_step(agt, env, s)

			if (step % self.steps_per_eval) == 0:
				for idx, agt in enumerate(agts):
					if self.hold_out_states is not None:
						if isinstance(agt, QLearning) or isinstance(agt, HeterogeneousMutualLearner):
							val = torch.mean(agt.get_action_vals(self.held_states))
						elif isinstance(agt, CartPoleADP):
							av_s = []
							for s in self.held_states:
								av_s.append(agt.get_action_vals(s.detach().numpy()))
							val = np.mean(av_s)
						eval_idx = int(np.floor(step / self.steps_per_eval))
						hist_idx = (idx, i, eval_idx)
						self.val_history[hist_idx] = val

				evals = [agt.evaluate(eval_env, 20) for (agt, eval_env) in zip(agts, eval_envs)]
				trial_evals.append(evals)
				print(f'Evals: {evals}')

		print(f'Trial {i}')
		return trial_evals

	def run(self, n_jobs=4):
		if n_jobs == 1:
			for trial in range(self.n_trials):
				trial_result = self._do_trial(trial)
				self.results.append(trial_result)
				self.step = 0

		else:
			with Pool(n_jobs) as p:
				results = p.map(self._do_trial, range(self.n_trials))

			self.results = results

	def _build_agts(self):
		agts = []
		if self.do_mutual:
			adp = CartPoleADP(
				nbins=self.adp_bins,
				gamma=self.gamma
			)
			qlrn = QLearning(
				gamma=self.gamma,
				initial_epsilon=self.initial_eps,
				final_epsilon=self.final_eps,
				epsilon_decay_steps=self.decay_steps,
				target_lag=self.q_target_lag,
				mutual_steps=self.mutual_steps,
				mutual_loss_weight = self.mutual_weight,
				mutual_loss_type = self.mutual_type
			)

			hook = MutHook(adp)

			qlrn.set_mutual_agents([adp])
			qlrn.set_mutual_hook(hook)

			adp.set_name('adp_sharing_tuples')
			qlrn.set_name('q_mutual')

			agts.append(adp)
			agts.append(qlrn)

		if self.do_standard_q:
			qlrn = QLearning(
				gamma=self.gamma,
				initial_epsilon=self.initial_eps,
				final_epsilon=self.final_eps,
				epsilon_decay_steps=self.decay_steps,
				target_lag=self.q_target_lag
			)

			qlrn.set_name('q_standard')
			agts.append(qlrn)

		if self.do_standard_adp:
			adp = CartPoleADP(
				nbins=self.adp_bins,
				gamma=self.gamma
			)

			adp.set_name('adp_standard')

			agts.append(adp)

		if self.do_hetmut:
			hetmut = HeterogeneousMutualLearner(
				mutual_steps=self.mutual_steps,
				initial_epsilon=self.initial_eps,
				final_epsilon=self.final_eps,
				epsilon_decay_steps=self.decay_steps,
				q_target_lag=self.q_target_lag,
				gamma=self.gamma,
				adp_bins=self.adp_bins
			)

			hetmut.set_name('single_agent_mutual')
			agts.append(hetmut)

		if self.names is None:
			names = [agt.name for agt in agts]
			print(f'Setting names to {names}')
			self.names = names

		return agts
