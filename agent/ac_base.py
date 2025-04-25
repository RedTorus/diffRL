import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.diffusion import Diffusion
from agent.vae       import VAE
from agent.model     import MLP
from agent.model import Critic

class ActorCriticBase:
    """
    Base class for QVPO, DiPo, and DDiffPG agents.
    Handles common actor/critic setup, optimizers, training loop, and utilities.
    Subclasses must implement:
      - append_memory(state, action, reward, next_state, mask)
      - compute_critic_loss(batch)
      - compute_actor_loss(batch)
    """
    def __init__(self, cfg, mode: str):
        self.cfg   = cfg
        self.mode  = mode
        self.step = 0
        # Core parameters
        self.state_dim    = cfg.state_dim
        self.action_space = cfg.action_space
        self.device       = cfg.device
        self.memory       = cfg.memory
        self.diffusion_memory = cfg.diffusion_memory

        # Hyperparameters
        self.batch_size  = cfg.args.batch_size
        self.num_steps   = cfg.args.num_steps
        self.policy_freq = cfg.args.policy_freq
        self.tau         = cfg.args.tau
        self.ac_grad_norm= cfg.args.ac_grad_norm
        self.update_actor_target_every = cfg.args.update_actor_target_every

        # Determine action dimension
        action_dim = int(torch.prod(torch.tensor(self.action_space.shape)))
        self.action_dim = action_dim
        # --- Actor setup ---

        policy_type = cfg.args.policy_type.lower()
        if policy_type == 'diffusion':
            self.actor = Diffusion(
                state_dim= self.state_dim,
                action_dim= action_dim,
                noise_ratio= cfg.args.noise_ratio,
                beta_schedule=cfg.args.beta_schedule,
                n_timesteps=cfg.args.n_timesteps,
                mode=mode
            ).to(self.device)
        elif policy_type == 'vae':
            self.actor = VAE(
                state_dim= self.state_dim,
                action_dim= action_dim,
                device= self.device
            ).to(self.device)
        else:
            self.actor = MLP(
                state_dim= self.state_dim,
                action_dim= action_dim
            ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        # --- Critic setup ---
        
        self.critic        = Critic(self.state_dim, action_dim, hidden_dim=512).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # --- Optimizers ---
        self.actor_optimizer  = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.args.diffusion_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.args.critic_lr, eps=1e-5
        )

        # Action scaling
        if self.action_space is None:
            self.action_scale = 1.
            self.action_bias  = 0.
        else:
            high = self.action_space.high
            low  = self.action_space.low
            self.action_scale = (high - low) / 2.0
            self.action_bias  = (high + low) / 2.0

        self.step = 0

    def soft_update(self, source, target):
        """
        Polyak averaging of target networks.
        """
        for p_src, p_tgt in zip(source.parameters(), target.parameters()):
            p_tgt.data.mul_(1 - self.tau)
            p_tgt.data.add_(self.tau * p_src.data)

    def optimizer_update(self, optimizer, loss):
        """
        Zero-grad, backpropagate, gradient clip, and optimizer step.
        """
        optimizer.zero_grad()
        loss.backward()
        if self.ac_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                max_norm=self.ac_grad_norm,
                norm_type=2
            )
        optimizer.step()

    def sample_action(self, state, eval=False):
        #dipo version
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        action = self.actor(state, eval).cpu().data.numpy().flatten()
        action = action.clip(-1, 1)
        action = action * self.action_scale + self.action_bias
        return action

    def train(self, log_callback=None):
        for _ in range(self.cfg.updates_per_step):
            # 1) sample batch, critic update (always)
            batch        = self.memory.sample(self.batch_size)
            critic_loss  = self.compute_critic_loss(batch)
            self.optimizer_update(self.critic_optimizer, critic_loss)

            # 2) actor update (diPo: every iter; qvpo: every policy_freq iters)
            do_actor = (
                self.mode == 'dipo'                 # DiPo always wants an update
                or (self.step % self.policy_freq == 0)  # QVPO wants it delayed
            )
            if do_actor:
                actor_loss = self.compute_actor_loss(batch)
                self.optimizer_update(self.actor_optimizer, actor_loss)
            else:
                actor_loss = None

            # 3) soft‐updates (same for both)
            self.soft_update(self.critic, self.critic_target)
            if self.step % self.update_actor_target_every == 0:
                self.soft_update(self.actor, self.actor_target)

            # 4) logging, step count, etc.
            if log_callback is not None:
                # if actor_loss is None, you can either skip it or log nan
                if actor_loss is None:
                    metrics = {
                        'critic_loss': critic_loss.item(),
                        'actor_loss': float('nan'),
                        'step': self.step
                    }
                else:
                    metrics = {
                        'critic_loss': critic_loss.item(),
                        'actor_loss':  actor_loss.item(),
                        'step': self.step
                    }
                log_callback(metrics, self.step)

            self.step += 1

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))

    # --- Algorithm-specific hooks ---
    def append_memory(self, state, action, reward, next_state, mask):
        raise NotImplementedError

    def compute_critic_loss(self, batch):
        raise NotImplementedError

    def compute_actor_loss(self, batch):
        raise NotImplementedError
    

