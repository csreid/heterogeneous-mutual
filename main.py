from itertools import product
import pickle
import time
import json
import argparse
import os

from experiment import Experiment

parser = argparse.ArgumentParser(
	description='Run an experiment'
)
parser.add_argument('fname')
parser.add_argument(
	'--n_jobs',
	help='Number of trials to run in parallel',
	default=1,
	type=int
)

def parse_config(fname):
	data = json.load(open(fname, 'r'))
	return data

if __name__ == '__main__':
	args = parser.parse_args()
	fname = args.fname
	n_jobs = args.n_jobs
	base = os.path.basename(fname)
	noext = os.path.splitext(base)[0]

	cfg = parse_config(fname)
	exp = Experiment(cfg)

	exp.run(n_jobs=n_jobs)
	exp.save(noext)
