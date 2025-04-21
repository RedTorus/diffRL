import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agent.ac_base import ActorCriticBase
from agent.helpers import EMA
from agent.q_transform import *

class QVPO(ActorCriticBase):
    def __init__(self, cfg):
        super().__init__(cfg, mode="qvpo")
        # QVPO-specific hyperparameters
        self.running_q_std = 1.0
        self.running_q_mean = 0.0
        self.beta = cfg.args.beta
        self.alpha_mean = cfg.args.alpha_mean
        self.alpha_std = cfg.args.alpha_std
        self.chosen = cfg.args.chosen
        self.q_neg = cfg.args.q_neg
        self.weighted = cfg.args.weighted
        self.aug = cfg.args.aug
        self.train_sample = cfg.args.train_sample
        self.q_transform = cfg.args.q_transform
        self.gradient = cfg.args.gradient
        self.policy_type = cfg.args.policy_type

    def append_memory(self, state, action, reward, next_state, mask):
        # scale action into [-1,1]
        action = (action - self.action_bias) / self.action_scale
        self.memory.append(state, action, reward, next_state, mask)
        if not self.aug:
            self.diffusion_memory.append(state, action)

    def sample_action(self, state, eval=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        normal = False
        if not eval and torch.rand(1).item() <= self.cfg.args.epsilon:
            normal = True
        action = self.actor(state, eval, q_func=self.critic, normal=normal)
        action = action.cpu().data.numpy().flatten().clip(-1, 1)
        return action * self.action_scale + self.action_bias

    def action_aug(self, batch_size, return_mean_std=False):
        states, actions, _, _, _ = self.memory.sample(batch_size)
        states, best_actions, v_pair, (mean, std) = self.actor.sample_n(
            states, times=self.train_sample, chosen=self.chosen,
            q_func=self.critic, origin=actions
        )
        if return_mean_std:
            return states, best_actions, (v_pair[0], v_pair[1]), (mean, std)
        return states, best_actions, (v_pair[0], v_pair[1])

    def action_gradient(self, batch_size, return_mean_std=False):
        states, best_actions, idxs = self.diffusion_memory.sample(batch_size)
        actions_optim = torch.optim.Adam([best_actions], lr=self.cfg.args.action_lr, eps=1e-5)
        for _ in range(self.cfg.args.action_gradient_steps):
            best_actions.requires_grad_(True)
            q1, q2 = self.critic(states, best_actions)
            loss = -torch.min(q1, q2).mean()
            actions_optim.zero_grad()
            loss.backward()
            if self.cfg.args.ratio > 0:
                nn.utils.clip_grad_norm_(
                    [best_actions], max_norm=self.action_dim * self.cfg.args.ratio
                )
            actions_optim.step()
            best_actions.requires_grad_(False)
            best_actions.clamp(-1., 1.)
        best_actions = best_actions.detach()
        self.diffusion_memory.replace(idxs, best_actions.cpu().numpy())
        if return_mean_std:
            q1, q2 = self.critic(states, best_actions)
            q = torch.min(q1, q2)
            return states, best_actions, (q.mean(), q.std())
        return states, best_actions

    def compute_critic_loss(self, batch):
        states, actions, rewards, next_states, masks = batch
        # current Q
        current_q1, current_q2 = self.critic(states, actions)
        next_actions = self.actor_target(next_states, eval=False, q_func=self.critic_target)
        # target Q
        target_q1, target_q2 = self.critic_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target_q = (rewards + masks * target_q).detach()
        return F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    def compute_actor_loss(self, batch):
        # get best actions and optional q-values
        if self.aug:
            if self.gradient:
                states, best_actions, qv, (mean, std) = self.action_gradient(self.batch_size, return_mean_std=True)
            else:
                states, best_actions, qv, (mean, std) = self.action_aug(self.batch_size, return_mean_std=True)
        else:
            states, best_actions = self.action_gradient(self.batch_size)
            v = None

        # only apply weighting for diffusion policies
        if self.policy_type.lower() == 'diffusion' and self.weighted:
            with torch.no_grad():
                q1, q2 = self.critic(states, best_actions)
                q = torch.min(q1, q2)
            # update running stats
            self.running_q_std += self.alpha_std * (std - self.running_q_std)
            self.running_q_mean += self.alpha_mean * (mean - self.running_q_mean)
            # extract v only when augmentation used
            if self.aug:
                _, v = qv
            # compute weights
            w = eval(self.q_transform)(
                q, q_neg=self.q_neg, cut=self.cfg.args.cut,
                running_q_std=self.running_q_std,
                running_q_mean=self.running_q_mean,
                beta=self.beta, v=v,
                batch_size=self.batch_size, chosen=self.chosen
            )
            return self.actor.loss(best_actions, states, weights=w)

        # default loss for non-diffusion or unweighted
        return self.actor.loss(best_actions, states)