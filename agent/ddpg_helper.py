# agent/ddpg_helpers.py

import random
import numpy as np
import torch

# ——— Replay Buffers —————————————————————————————

class NStepReplay:
    def __init__(self, capacity, nstep, gamma):
        self.capacity = capacity
        self.nstep    = nstep
        self.gamma    = gamma
        self.buffer   = []

    def add(self, state, action, reward, next_state, mask):
        self.buffer.append((state, action, reward, next_state, mask))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # transpose to five lists...
        return tuple(map(np.stack, zip(*batch)))

class DiffusionGoalBuffer:
    def __init__(self, capacity, action_dim, behavior_sample):
        self.capacity       = capacity
        self.action_dim     = action_dim
        self.behavior_sample= behavior_sample
        self.states         = []
        self.actions        = []

    def append(self, state, action):
        self.states.append(state)
        self.actions.append(action)
        if len(self.states) > self.capacity:
            self.states.pop(0); self.actions.pop(0)

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self.states), batch_size)
        s = torch.FloatTensor([self.states[i]  for i in idxs])
        a = torch.FloatTensor([self.actions[i] for i in idxs])
        return s, a, idxs

    def replace(self, idxs, new_actions):
        for i, a in zip(idxs, new_actions):
            self.actions[i] = a

# ——— Schedulers & Noise —————————————————————————

class LinearSchedule:
    def __init__(self, start, end, decay):
        self.start = start
        self.end   = end
        self.decay = decay

    def value(self, step):
        frac = min(step / self.decay, 1.0)
        return self.start + frac * (self.end - self.start)

def add_normal_noise(actions, std):
    noise = torch.randn_like(actions) * std
    return (actions + noise).clamp(-1, 1)

# ——— Miscellaneous ————————————————————————————

def soft_update(source, target, tau):
    for p_src, p_tgt in zip(source.parameters(), target.parameters()):
        p_tgt.data.mul_(1 - tau)
        p_tgt.data.add_(tau * p_src.data)
