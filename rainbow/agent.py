# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from .model import DQN, ImprovedDQN


class Agent():
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip
    self.image_only = args.image_only

    self.online_net = DQN(args, self.action_space).to(device=args.device) if not args.use_deep_conv else ImprovedDQN(args, self.action_space).to(device=args.device)
    if args.model:
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]
            del state_dict[old_key]
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      else:
        raise FileNotFoundError(args.model)

    self.online_net.train()

    self.target_net = DQN(args, self.action_space).to(device=args.device) if not args.use_deep_conv else ImprovedDQN(args, self.action_space).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  def reset_noise(self):
    self.online_net.reset_noise()

  def act(self, state):
    with torch.no_grad():
      if self.image_only:
        state_batch = {
          "image": state["image"].unsqueeze(0),
        }
      else:
        state_batch = {
          "image": state["image"].unsqueeze(0),
          "sensor": state["sensor"].unsqueeze(0)
        }
      return (self.online_net(state_batch) * self.support).sum(2).argmax(1).item()

  def act_e_greedy(self, state, epsilon=0.001):
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem):
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    log_ps = self.online_net(states, log=True)
    log_ps_a = log_ps[range(self.batch_size), actions]

    with torch.no_grad():
      pns = self.online_net(next_states)
      dns = self.support.expand_as(pns) * pns
      argmax_indices_ns = dns.sum(2).argmax(1)
      self.target_net.reset_noise()
      pns = self.target_net(next_states)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
      b = (Tz - self.Vmin) / self.delta_z
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      m = states["image"].new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))

    loss = -torch.sum(m * log_ps_a, 1)
    self.online_net.zero_grad()
    (weights * loss).mean().backward()
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())
    return loss

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  def evaluate_q(self, state):
    with torch.no_grad():
      if self.image_only:
        state_batch = {
          "image": state["image"].unsqueeze(0),
        }
      else:
        state_batch = {
          "image": state["image"].unsqueeze(0),
          "sensor": state["sensor"].unsqueeze(0)
        }
      return (self.online_net(state_batch) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
