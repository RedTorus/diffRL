# agent/ddiffpg.py

import copy
import torch
import torch.nn.functional as F
import numpy as np

from agent.ac_base        import ActorCriticBase
from agent.diffusion      import Diffusion
from agent.model          import Critic
from agent.ddpg_helper   import (
    NStepReplay,
    DiffusionGoalBuffer,
    LinearSchedule,
    add_normal_noise,
    soft_update
)

class DDiffPG(ActorCriticBase):
    def __init__(self, cfg):
        super().__init__(cfg, mode="ddiffpg")

        state_dim  = cfg.state_dim
        action_dim = int(np.prod(cfg.action_space.shape))

        # — Actor (diffusion policy) —
        self.actor = Diffusion(
            state_dim     = state_dim,
            action_dim    = action_dim,
            noise_ratio   = cfg.args.noise_std_max,
            beta_schedule = cfg.args.beta_schedule,
            n_timesteps   = cfg.args.n_timesteps,
            mode          = "ddiffpg"
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        # — Critic setup —
        self.critic        = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # — Replay buffers —
        self.memory           = NStepReplay(
            capacity = cfg.args.memory_size,
            nstep    = cfg.args.nstep,
            gamma    = cfg.gamma
        )
        self.diffusion_memory = DiffusionGoalBuffer(
            capacity        = cfg.args.diffusion_memory_size,
            action_dim      = action_dim,
            behavior_sample = cfg.args.behavior_sample
        )

        # — Noise scheduler —
        self.noise_scheduler = LinearSchedule(
            start = cfg.args.noise_std_max,
            end   = cfg.args.noise_std_min,
            decay = cfg.args.noise_decay_steps
        )

        # — Hyperparameters —
        self.batch_size  = cfg.args.batch_size
        self.gamma       = cfg.args.gamma
        self.tau         = cfg.args.tau
        self.policy_freq = cfg.args.policy_freq
        self.learn_steps = 0

        # — Optimizers —
        self.actor_optimizer  = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.args.actor_lr,  eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.args.critic_lr, eps=1e-5
        )

    def append_memory(self, state, action, reward, next_state, mask):
        # scale action into [-1,1] before storing
        a = (action - self.action_bias) / self.action_scale
        self.memory.add(state, a, reward, next_state, mask)
        self.diffusion_memory.append(state, a)

    def sample_action(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # configure reverse chain length for evaluation
        self.actor.set_timesteps(self.cfg.args.eval_steps, device=self.device)
        with torch.no_grad():
            raw = self.actor(state, eval)
        act = raw.cpu().numpy().flatten().clip(-1,1)
        return act * self.action_scale + self.action_bias

    def train(self, t, iterations):
        for jj in range(iterations):
            # 1) Critic update
            batch  = self.memory.sample(self.batch_size)
            
            loss_c = self._compute_critic_loss(batch)
            self.critic_optimizer.zero_grad()
            loss_c.backward()
            self.critic_optimizer.step()
            print("Iteration; ", jj)
            # 2) Actor update (delayed)
            if self.learn_steps % self.policy_freq == 0:
                loss_a = self._compute_actor_loss(batch)
                self.actor_optimizer.zero_grad()
                loss_a.backward()
                self.actor_optimizer.step()

            # 3) Soft target updates
            soft_update(self.critic,        self.critic_target, self.tau)
            soft_update(self.actor,         self.actor_target,  self.tau)

            self.learn_steps += 1
        print("training done")

    def _compute_critic_loss(self, batch):
        # Unpack batch (NumPy arrays)
        states, actions, rewards, next_states, masks = batch

        # Convert to torch tensors on self.device
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.FloatTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        masks       = torch.FloatTensor(masks).to(self.device).unsqueeze(-1)

        # Critic target Q
        with torch.no_grad():
            self.actor_target.set_timesteps(self.batch_size, device=self.device)
            next_actions = self.actor_target.forward(next_states, eval=False)
            q1_t, q2_t = self.critic_target(next_states, next_actions)
            q_target   = torch.min(q1_t, q2_t)
            target     = rewards + masks * self.gamma * q_target

        # Current Q estimates
        q1, q2 = self.critic(states, actions)
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

    def _compute_actor_loss(self, batch):
        # Unpack batch (states only)
        states, _, _, _, _ = batch

        # Convert to torch tensor on self.device
        states = torch.FloatTensor(states).to(self.device)

        # Sample actions from diffusion actor and add exploration noise
        a0  = self.actor.forward(states, eval=False)
        std = self.noise_scheduler.value(self.learn_steps)
        ano = add_normal_noise(a0, std)

        # Diffusion actor loss
        return self.actor.loss(ano, states)

    def save_model(self, dir, id=None):
        super().save_model(dir, id)

    def load_model(self, dir, id=None):
        super().load_model(dir, id)
