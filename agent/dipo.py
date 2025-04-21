import torch
import numpy as np
import copy
import torch.nn as nn
from agent.ac_base import ActorCriticBase
import torch.nn.functional as F
from agent.diffusion import Diffusion

class DiPo(ActorCriticBase):
    def __init__(self, cfg):
        super().__init__(cfg, mode="dipo")
        # DiPo-specific hyperparameters
        self.action_gradient_steps = cfg.args.action_gradient_steps
        self.action_grad_norm = self.action_dim * cfg.args.ratio
        self.action_lr = cfg.args.action_lr

    def append_memory(self, state, action, reward, next_state, mask):
        action = (action - self.action_bias) / self.action_scale
        self.memory.append(state, action, reward, next_state, mask)
        self.diffusion_memory.append(state, action)

    def action_gradient(self, batch_size):
        states, best_actions, idxs = self.diffusion_memory.sample(batch_size)

        actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)


        for i in range(self.action_gradient_steps):
            best_actions.requires_grad_(True)
            q1, q2 = self.critic(states, best_actions)
            loss = -torch.min(q1, q2)

            actions_optim.zero_grad()

            loss.backward(torch.ones_like(loss))
            if self.action_grad_norm > 0:
                actions_grad_norms = nn.utils.clip_grad_norm_([best_actions], max_norm=self.action_grad_norm, norm_type=2)

            actions_optim.step()

            best_actions.requires_grad_(False)
            best_actions.clamp_(-1., 1.)

        # if self.step % 10 == 0:
        #     log_writer.add_scalar('Action Grad Norm', actions_grad_norms.max().item(), self.step)

        best_actions = best_actions.detach()

        self.diffusion_memory.replace(idxs, best_actions.cpu().numpy())

        return states, best_actions

    def compute_critic_loss(self, batch):
        states, actions, rewards, next_states, masks = batch
        current_q1, current_q2 = self.critic(states, actions)
        next_actions = self.actor_target(next_states, eval=False)
        target_q1, target_q2 = self.critic_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target_q = (rewards + masks * target_q).detach()
        return F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    def compute_actor_loss(self, batch):
        # refine actions via gradient ascent
        states, best_actions = self.action_gradient(
            self.cfg.args.batch_size
        )
        # supervised diffusion/VAE loss
        loss = self.actor.loss(best_actions, states)
        return loss
