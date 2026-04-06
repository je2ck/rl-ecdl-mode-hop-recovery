# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle

import numpy as np
import torch
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from .agent import Agent
from .env import Env
from .memory import ReplayMemory
from .test import test
from .logging_utils import create_loggers


parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0, type=int, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
parser.add_argument('--model-current-only', action='store_true', help='Use model that controls only current')
parser.add_argument('--target-frequency', type=float, default=751.52630, help='target frequency')
parser.add_argument('--random-target', action='store_true', help='Use random target frequency')
parser.add_argument('--simulation', action='store_true', help='Use laser simulator')
parser.add_argument('--image-only', action='store_true', help='Use image data only')
parser.add_argument('--frame-skip-num', type=int, default=4, help='Number of frames to skip')
parser.add_argument('--current-range', type=str, default='short', help='current range long or short')
parser.add_argument('--render-with-gradcam', action='store_true', help='enable gradcam when rendering')
parser.add_argument('--use-deep-conv', action='store_true', help='use improved dqn')
parser.add_argument('--validation', action='store_true', help='validation')

# Setup
args = parser.parse_args()

writer = SummaryWriter(log_dir="runs/" + args.id)

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('results', args.id)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')


def log(s):
  with open(os.path.join(results_dir, 'log.txt'), 'a') as log_file:
    log_file.write('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s + '\n')
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)


# Environment
env = Env(args)
env.train()
action_space = env.action_space()

# Agent
dqn = Agent(args, env)

if args.model is not None and not args.evaluate:
  if not args.memory:
    raise ValueError('Cannot resume training without memory save path. Aborting...')
  elif not os.path.exists(args.memory):
    raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

  mem = load_memory(args.memory, args.disable_bzip_memory)

else:
  mem = ReplayMemory(args, args.memory_capacity)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
  if done:
    state = env.reset()

  next_state, _, done = env.step(np.random.randint(0, action_space))
  val_mem.append(state, -1, 0.0, done)
  state = next_state
  T += 1

if args.evaluate:
  dqn.eval()
  avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
  # Training loop
  dqn.train()
  done = True
  episode_reward = 0
  episode_count = 0
  episode_step = 0
  episode_rewards = []

  episode_logger, evaluation_logger, streaming_logger = create_loggers(results_dir)

  for T in trange(1, args.T_max + 1):
    if done:
      if T > 1:
        episode_info = {
          'episode': episode_count,
          'reward': episode_reward,
          'steps': episode_step,
          'training_step': T,
          'timestamp': datetime.now().isoformat()
        }
        episode_rewards.append(episode_reward)

        episode_logger.log_episode(episode_info)
        streaming_logger.log_episode(episode_info)

        writer.add_scalar('Episode/reward', episode_reward, episode_count)
        writer.add_scalar('Episode/steps', episode_step, episode_count)
        writer.add_scalar('Episode/reward_by_training_step', episode_reward, T)

        episode_count += 1

      state = env.reset()
      episode_reward = 0
      episode_step = 0

    if T % args.replay_frequency == 0:
      dqn.reset_noise()

    action = dqn.act(state)
    next_state, reward, done = env.step(action)
    episode_reward += reward
    episode_step += 1

    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)
    mem.append(state, action, reward, done)

    if T >= args.learn_start:
      mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

      if T % args.replay_frequency == 0:
        loss = dqn.learn(mem)
        if loss is not None:
          writer.add_scalar('Loss/train', loss.mean(), T)

      if T % args.evaluation_interval == 0:
        dqn.eval()
        avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir)
        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))

        writer.add_scalar('Reward/avg_reward', avg_reward, T)
        writer.add_scalar('Q/avg_Q', avg_Q, T)
        dqn.train()

        if args.memory is not None:
          save_memory(mem, args.memory, args.disable_bzip_memory)

      if T % args.target_update == 0:
        dqn.update_target_net()

      if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
        dqn.save(results_dir, 'checkpoint.pth')

    state = next_state

streaming_logger.close()

import json
final_summary = {
  'total_episodes': episode_count,
  'total_training_steps': args.T_max,
  'final_episode_rewards': episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards,
  'training_completed': datetime.now().isoformat()
}
with open(os.path.join(results_dir, 'training_summary.json'), 'w') as f:
  json.dump(final_summary, f, indent=2)

print(f"Training completed! Total episodes: {episode_count}")

env.close()
writer.close()
