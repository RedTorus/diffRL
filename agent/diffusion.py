import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agent.helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)

from agent.model import Model


class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, noise_ratio,
                 beta_schedule='vp', n_timesteps=1000,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True,
                 behavior_sample=16, eval_sample=512, deterministic=False, mode='qvpo', 
                 diffusion_mode='ddpm', num_inference_steps=0, order_k=4):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = Model(state_dim, action_dim, hidden_size=256)
        self.mode = mode #### for different agents
        self.max_noise_ratio = noise_ratio
        self.noise_ratio = noise_ratio

        self.behavior_sample = behavior_sample
        self.eval_sample = eval_sample
        self.deterministic = deterministic

        self.diffusion_mode = diffusion_mode
        self.num_inference_steps = num_inference_steps
        if num_inference_steps == 0:
            if diffusion_mode == 'ddpm':
                self.num_inference_steps = n_timesteps
            elif diffusion_mode == 'ddim':
                self.num_inference_steps = n_timesteps
            elif diffusion_mode == 'ddim_stochastic':
                self.num_inference_steps = n_timesteps
            elif diffusion_mode == 'lms':
                self.num_inference_steps = n_timesteps
            elif diffusion_mode == 'rk':
                self.num_inference_steps = n_timesteps
            elif diffusion_mode == 'dpmsolver':
                self.num_inference_steps = n_timesteps
            elif diffusion_mode == 'heun':
                self.num_inference_steps = n_timesteps
            elif diffusion_mode == 'pc':
                self.num_inference_steps = n_timesteps
            else:
                raise ValueError(f"Unknown diffusion_mode={self.diffusion_mode}")

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()
        self.eta = 0.5  # eta for DDIM sampling
        self.order_k = order_k
        self.lms_k = min(self.order_k, 8)  # max order of Adams-Bashforth
        self.rk_k = min(self.order_k, 4)  # max order of Runge-Kutta
        self.dpm_solver_order = min(self.order_k, 3)  # max order of DPM-Solver
        self.adams_bashforth_coeffs = {
            1: [1.0],
            2: [3/2, -1/2],
            3: [23/12, -16/12, 5/12],
            4: [55/24, -59/24, 37/24, -9/24],          # ← PNDM
            5: [1901/720, -2774/720, 2616/720, -1274/720, 251/720],
            6: [4277/1440, -7923/1440, 9982/1440, -7298/1440, 2877/1440, -475/1440],
            7: [198721/60480, -447288/60480, 705549/60480,
                -688256/60480, 407139/60480, -134472/60480, 19087/60480],
            8: [434241/120960, -1152169/120960, 2183877/120960,
                -2664477/120960, 2102243/120960, -1041723/120960,
                295767/120960, -36799/120960],
        }


    def set_timesteps(self, device=None):
        """
        Build a reversed list of timesteps for inference based on num_inference_steps.
        """
        timesteps = torch.linspace(
            0,
            self.n_timesteps - 1,
            self.num_inference_steps,
            device=device or self.betas.device
        ).long()
        self.timesteps = timesteps.flip(0)


    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly

            Given a noisy sample x_t at timestep t and either:
            - `noise = εθ(x_t, t)` if predict_epsilon=True, or
            - `noise = x̂₀` if predict_epsilon=False,
            recover the model’s estimate of x₀:
            
            If predicting ε:
                x̂₀ = (x_t / sqrt(ᾱ_t)) - (√(1−ᾱ_t)/√ᾱ_t) · ε
            
            else:
                the model already outputs x̂₀ directly.
        '''
        if self.predict_epsilon:
            #compute the estimate  x0 ​of the original clean sample
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, s):
        #Performs one ancestral reverse step of DDPM
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise * self.noise_ratio


    @torch.no_grad()
    def p_sample_loop(self, state, shape):
        """
        DDPM ancestral sampling over self.timesteps for uniform inference steps.
        ancestral DDPM sampling for all selected timesteps, starting from pure Gaussian noise at t=T and iterating downward to t=0
        """
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        # use uniform timesteps for DDPM as well
        self.set_timesteps(device=device)
        for t in self.timesteps:
            ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, ts, state)
        return x

    # -------------------- New Samplers --------------------
    
    @torch.no_grad()
    def _ddim_step(self, x, t, next_t, state, stochastic: bool):
        """
        One reverse‐DDIM step, optionally with stochastic noise.
        
        Args:
        x        : current x_t, shape (B, action_dim)
        t        : tensor of shape (B,) holding the current timestep
        next_t   : tensor of shape (B,) holding the next timestep (or 0)
        state    : conditioning state passed to your model
        stochastic: if True, inject noise scaled by eta*noise_ratio
        
        Returns:
        x_{t-1} estimate, shape (B, action_dim)
        """
        # 1) predict εθ(x_t,t) or x₀ directly
        eps_or_x0 = self.model(x, t, state)
        x0 = self.predict_start_from_noise(x, t, eps_or_x0)

        # 2) optional clamp of x0
        if self.clip_denoised:
            x0 = x0.clamp(-1., 1.)

        # 3) fetch ᾱ_t and ᾱ_{t-1}
        a_t     = extract(self.alphas_cumprod,     t,      x.shape)
        a_prev  = extract(self.alphas_cumprod, next_t,   x.shape)
        sqrt_a_t    = a_t.sqrt()
        sqrt_a_prev = a_prev.sqrt()

        # 4) deterministic DDIM drift
        dir_xt = (sqrt_a_prev * x0 - sqrt_a_t * x) / sqrt_a_t
        x_prev = sqrt_a_prev * x0 + dir_xt

        # 5) optional stochastic
        if stochastic:
            # compute σ_t = η·noise_ratio·√((1−ᾱ_{t-1})/(1−ᾱ_t))·√(1−ᾱ_t/ᾱ_{t-1})
            sigma_t = (
                self.eta
                * self.noise_ratio
                * ((1 - a_prev) / (1 - a_t)).sqrt()
                * (1 - a_t / a_prev).sqrt()
            )
            noise = torch.randn_like(x)
            mask  = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
            x_prev = x_prev + mask * sigma_t * noise

        return x_prev

    @torch.no_grad()
    def _dpm_solver_step(self, x, ts, nts, state):
        """
        One step of DPM-Solver-2 or -3:
        - ts  = current timestep tensor
        - nts = next   timestep tensor
        - x   = x_t
        - state = conditioning
        Returns x_{t-1}.
        """
        # 1) Compute alpha_i, alpha_j and h = ln(alpha_j/alpha_i)
        alpha_i = extract(self.alphas_cumprod, ts,  x.shape)
        alpha_j = extract(self.alphas_cumprod, nts, x.shape)
        h       = alpha_j.log() - alpha_i.log()

        # 2) Predictor at i: ε_i and x0_i
        eps_i = self.model(x, ts, state)
        x0_i  = self.predict_start_from_noise(x, ts, eps_i)

        # 3) Exponential–Euler predictor to j
        sqrt_alpha_j            = alpha_j.sqrt()
        sqrt_one_minus_alpha_j  = extract(self.sqrt_one_minus_alphas_cumprod, nts, x.shape)
        x_j_e                   = sqrt_alpha_j * x0_i + sqrt_one_minus_alpha_j * eps_i

        # 4) Corrector eval at j: ε_j and x0_j
        eps_j = self.model(x_j_e, nts, state)
        x0_j  = self.predict_start_from_noise(x_j_e, nts, eps_j)

        # 5) 2nd-order update
        eta = (h.exp() - 1) / h
        if self.dpm_solver_order == 2:
            return sqrt_alpha_j * (eta * x0_i + (1 - eta) * x0_j)

        # 6) 3rd-order midpoint
        alpha_mid = (alpha_i.log() + 0.5 * h).exp()
        sqrt_alpha_mid           = alpha_mid.sqrt()
        sqrt_one_minus_alpha_mid = (1 - alpha_mid).sqrt().unsqueeze(-1)
        x_mid_e                  = sqrt_alpha_mid * x0_i + sqrt_one_minus_alpha_mid * eps_i

        eps_mid = self.model(x_mid_e, ts, state)
        x0_mid  = self.predict_start_from_noise(x_mid_e, ts, eps_mid)

        # 7) 3rd-order weights
        b0 = ((h.exp() - 1)*(h - 2) + 2*(h.exp() - 1)) / (2 * h * h)
        b1 = (2*h - 4 + 2*h.exp()*(2 - h)) / (2 * h * h)

        # 8) Combine for order-3
        return sqrt_alpha_j * (b0 * x0_i + b1 * x0_mid + b0 * x0_j)

    @torch.no_grad()
    def ddim_sample(self, state):
        """Purely deterministic DDIM (η=0)."""
        device = self.betas.device
        B = state.shape[0]
        x = torch.randn((B, self.action_dim), device=device)
        self.set_timesteps(device=device)

        for idx, t in enumerate(self.timesteps):
            ts     = torch.full((B,), t,      device=device, dtype=torch.long)
            next_t = (self.timesteps[idx+1]
                    if idx+1 < len(self.timesteps) else 0)
            nts    = torch.full((B,), next_t, device=device, dtype=torch.long)

            x = self._ddim_step(x, ts, nts, state, stochastic=False)

        return x.clamp(-1., 1.)


    @torch.no_grad()
    def ddim_sample_stochastic(self, state):
        """DDIM with per-step noise (η>0)."""
        device = self.betas.device
        B = state.shape[0]
        x = torch.randn((B, self.action_dim), device=device)
        self.set_timesteps(device=device)

        for idx, t in enumerate(self.timesteps):
            ts     = torch.full((B,), t,      device=device, dtype=torch.long)
            next_t = (self.timesteps[idx+1]
                    if idx+1 < len(self.timesteps) else 0)
            nts    = torch.full((B,), next_t, device=device, dtype=torch.long)

            x = self._ddim_step(x, ts, nts, state, stochastic=True)

        return x.clamp(-1., 1.)

    @torch.no_grad()
    def lms_sample(self, state):
        """
        General k-step Linear Multistep (Adams–Bashforth) sampler,
        order = self.lms_k (max 8).  PNDM is simply k=4.
        """
        device = self.betas.device
        B = state.shape[0]
        x = torch.randn((B, self.action_dim), device=device)
        self.set_timesteps(device=device)

        old_derivs = []
        for idx, t in enumerate(self.timesteps):
            # pack current & next timestep
            ts     = torch.full((B,), t,      device=device, dtype=torch.long)
            next_t = (self.timesteps[idx+1]
                      if idx+1 < len(self.timesteps) else 0)
            nts    = torch.full((B,), next_t, device=device, dtype=torch.long)

            # predict x0 and compute derivative f_i = (x - x0)/σ_t
            eps   = self.model(x, ts, state)
            x0    = self.predict_start_from_noise(x, ts, eps)
            sigma = extract(self.sqrt_recipm1_alphas_cumprod, ts, x.shape)
            f_i   = (x - x0) / sigma

            # update derivative buffer
            old_derivs.append(f_i)
            if len(old_derivs) > self.lms_k:
                old_derivs.pop(0)

            # step size h = σ_{i+1} - σ_i
            sigma_next = extract(self.sqrt_recipm1_alphas_cumprod, nts, x.shape)
            h = sigma_next - sigma

            # determine how many past steps we actually have
            order = min(len(old_derivs), self.lms_k)
            coeffs = self.adams_bashforth_coeffs[order]

            # form the weighted sum of the last `order` derivatives
            deriv_terms = [
                coeffs[j] * old_derivs[-1 - j]
                for j in range(order)
            ]
            deriv_combined = sum(deriv_terms)

            # advance along the deterministic ODE
            x = x + h * deriv_combined

        return x.clamp(-1., 1.)

    @torch.no_grad()
    def rk_sample(self, state):
        """
        k-stage explicit Runge–Kutta sampler over the σ-ODE.
        Uses k stages per step (self.rk_k in {2,3,4}).
        """
        device = self.betas.device
        B = state.shape[0]
        x = torch.randn((B, self.action_dim), device=device)
        self.set_timesteps(device=device)

        # Butcher tableaus for supported k
        rk_params = {
            2: {  # Midpoint (RK2)
                "c":[0, 0.5],
                "a":[[], [0.5]],
                "b":[0, 1]  # note: normalized to sum=1
            },
            3: {  # Heun’s method (RK3)
                "c":[0, 0.5, 1],
                "a":[[],
                     [0.5],
                     [-1, 2]],
                "b":[1/6, 2/3, 1/6]
            },
            4: {  # Classical RK4
                "c":[0, 0.5, 0.5, 1],
                "a":[[],
                     [0.5],
                     [0, 0.5],
                     [0, 0, 1]],
                "b":[1/6, 1/3, 1/3, 1/6]
            },
        }[self.rk_k]

        for idx, t in enumerate(self.timesteps):
            ts     = torch.full((B,), t,      device=device, dtype=torch.long)
            next_t = (self.timesteps[idx+1]
                      if idx+1 < len(self.timesteps) else 0)
            nts    = torch.full((B,), next_t, device=device, dtype=torch.long)

            # get σ_i, σ_{i+1}
            sigma_i     = extract(self.sqrt_recipm1_alphas_cumprod, ts,  x.shape)
            sigma_next  = extract(self.sqrt_recipm1_alphas_cumprod, nts, x.shape)
            h = sigma_next - sigma_i

            # pre-allocate stages
            ks = []

            for stage in range(self.rk_k):
                c = rk_params["c"][stage]
                # build σ_stage = σ_i + c*h
                sigma_stage = sigma_i + c * h

                # build x_stage = x + h * sum_j (a[stage][j] * ks[j])
                if stage == 0:
                    x_stage = x
                else:
                    acc = torch.zeros_like(x)
                    for j, a_ij in enumerate(rk_params["a"][stage]):
                        acc = acc + a_ij * ks[j]
                    x_stage = x + h * acc

                # compute f = (x_stage - x0_stage) / σ_stage
                # need corresponding t_stage: pick nearest
                # (simplest: use ts for all stages, since \hat x0 only weakly dependent on t)
                eps = self.model(x_stage, ts, state)
                x0  = self.predict_start_from_noise(x_stage, ts, eps)
                f   = (x_stage - x0) / sigma_stage
                ks.append(f)

            # combine stages: x_{i+1} = x + h*sum(b_j * k_j)
            update = sum(b * k for b, k in zip(rk_params["b"], ks))
            x = x + h * update

        return x.clamp(-1., 1.)


    @torch.no_grad()
    def dpm_solver_sample(self, state):
        """
        DPM-Solver (order 2 or 3) using a private per-step helper.
        """
        device = self.betas.device
        B = state.shape[0]
        x = torch.randn((B, self.action_dim), device=device)
        self.set_timesteps(device=device)

        for idx, t in enumerate(self.timesteps):
            ts     = torch.full((B,), t,      device=device, dtype=torch.long)
            next_t = (self.timesteps[idx+1] if idx+1 < len(self.timesteps) else 0)
            nts    = torch.full((B,), next_t, device=device, dtype=torch.long)
            x = self._dpm_solver_step(x, ts, nts, state)

        return x.clamp(-1., 1.)

    @torch.no_grad()
    def heun_sde_sample(self, state):
        """
        Heun’s two‐stage SDE integrator (strong order 1):
            1) Predictor: Euler–Maruyama
            2) Corrector: trapezoidal update
        """
        device = self.betas.device
        B = state.shape[0]
        x = torch.randn((B, self.action_dim), device=device)
        self.set_timesteps(device=device)

        for idx, t in enumerate(self.timesteps):
            # current & next timestep indices
            ts     = torch.full((B,), t,      device=device, dtype=torch.long)
            next_t = self.timesteps[idx+1] if idx+1 < len(self.timesteps) else 0
            nts    = torch.full((B,), next_t, device=device, dtype=torch.long)

            # — Predictor (Euler–Maruyama) —
            # compute model mean and log‐variance at (x_t, t)
            model_mean_t, _, model_log_var_t = self.p_mean_variance(x=x, t=ts, s=state)
            # noise standard deviation scaled by noise_ratio
            std_t = (0.5 * model_log_var_t).exp() * self.noise_ratio
            # draw one noise sample for both stages
            z = torch.randn_like(x)

            # build noise‐mask so we don't re‐noise at t==0
            nonzero = (t != 0).float().view(B, *([1] * (x.ndim - 1)))

            # Euler step
            x_e = model_mean_t + nonzero*std_t * z

            # — Corrector (Heun) —
            # recompute mean at the Euler‐predicted point (x_e, t−Δt)
            model_mean_e, _, _ = self.p_mean_variance(x=x_e, t=nts, s=state)
            # trapezoidal update + same noise term
            x = x + 0.5 * ((model_mean_t + model_mean_e) - x) + std_t * z

        return x.clamp(-1., 1.)

    @torch.no_grad()
    def pc_sampler(self, state, snr=0.16, n_corrector=1):
        """
        Predictor–Corrector sampler:
          - n_corrector Langevin (score‐based) steps per diffusion step
          - 1 ancestral SDE step
        """
        device = self.betas.device
        B = state.shape[0]
        x = torch.randn((B, self.action_dim), device=device)
        self.set_timesteps(device=device)

        for t in self.timesteps:
            ts = torch.full((B,), t, device=device, dtype=torch.long)

            # Precompute σ_t = sqrt((1−ᾱ_t)/ᾱ_t)
            alpha_bar = extract(self.alphas_cumprod, ts, x.shape)
            sigma_t   = (1 - alpha_bar).sqrt() / alpha_bar.sqrt()

            # — Corrector: n_corrector steps of Langevin MCMC —
            for _ in range(n_corrector):
                eps = self.model(x, ts, state)                 # εθ(x,t)
                score = -eps / sigma_t.unsqueeze(-1)           # ∇ log p ≈ −ε/σ
                # compute step‐size via SNR heuristic
                grad_norm  = score.flatten(1).norm(dim=-1).mean()
                noise_norm = torch.randn_like(x).flatten(1).norm(dim=-1).mean()
                step_size  = (snr * noise_norm / grad_norm) ** 2 * 2
                z = torch.randn_like(x)
                x = x + step_size * score + (2 * step_size) ** 0.5 * z

            # — Predictor: standard DDPM ancestral step —
            model_mean, _, model_log_var = self.p_mean_variance(x=x, t=ts, s=state)
            noise = torch.randn_like(x)
            # build noise‐mask so we don't re‐noise at t==0
            nonzero = (t != 0).float().view(B, *([1] * (x.ndim - 1)))
            x = model_mean + nonzero*(noise * (0.5 * model_log_var).exp() * self.noise_ratio)

        return x.clamp(-1., 1.)

    def _call_sampler(self, s, shape):
        if self.diffusion_mode == 'ddpm':
            return self.p_sample_loop(s, shape)
        elif self.diffusion_mode == 'ddim':
            return self.ddim_sample(s)
        elif self.diffusion_mode == 'ddim_stochastic':
            return self.ddim_sample_stochastic(s)
        elif self.diffusion_mode == 'lms':
            return self.lms_sample(s)
        elif self.diffusion_mode == 'rk':
            return self.rk_sample(s)
        elif self.diffusion_mode == 'dpmsolver':
            return self.dpm_solver_sample(s)
        elif self.diffusion_mode == 'heun':
            return self.heun_sde_sample(s)
        elif self.diffusion_mode == 'pc':
            return self.pc_sampler(s)
        else:
            raise ValueError(f"Unknown diffusion_mode={self.diffusion_mode}")

    @torch.no_grad()
    def sample(self, state, eval=False, q_func=None, normal=False):
        # — noise_ratio logic unchanged —
        if self.mode == 'qvpo':
            if self.deterministic:
                self.noise_ratio = 0 if eval else self.max_noise_ratio
            else:
                self.noise_ratio = self.max_noise_ratio

            # normal‐mode shortcut
            if normal:
                batch_size = state.shape[0]
                shape = (batch_size, self.action_dim)
                action = self._call_sampler(state, shape)
                action.clamp_(-1., 1.)
                return action

            # best‐of‐N in eval vs behavior
            raw_batch_size = state.shape[0]
            if eval:
                reps = self.eval_sample
            else:
                reps = self.behavior_sample

            state_rep = state.repeat(reps, 1)
            shape = (state_rep.shape[0], self.action_dim)
            action = self._call_sampler(state_rep, shape)
            action.clamp_(-1., 1.)

            q1, q2 = q_func(state_rep, action)
            q = torch.min(q1, q2)
            action = action.view(reps, raw_batch_size, -1).transpose(0, 1)
            q      = q.view     (reps, raw_batch_size, -1).transpose(0, 1)
            idx = torch.argmax(q, dim=1, keepdim=True).repeat(1, 1, self.action_dim)
            return action.gather(dim=1, index=idx).view(raw_batch_size, -1)

        elif self.mode == 'dipo':
            self.noise_ratio = 0 if eval else self.max_noise_ratio
            batch_size = state.shape[0]
            shape = (batch_size, self.action_dim)
            action = self._call_sampler(state, shape)
            return action.clamp(-1., 1.)

        elif self.mode == 'ddiffpg':
            self.noise_ratio = 0 if eval else self.max_noise_ratio
            batch_size = state.shape[0]
            shape = (batch_size, self.action_dim)
            action = self._call_sampler(state, shape)
            return action.clamp(-1., 1.)

        else:
            raise ValueError(f"Unknown mode={self.mode}")



    @torch.no_grad()
    def dipo_sample(self, state, eval=False):
        self.noise_ratio = 0 if eval else self.max_noise_ratio
        
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape)
        return action.clamp_(-1., 1.)

    # ------------------------------------------ training ------------------------------------------#
    @torch.no_grad()
    def sample_n(self, state, eval=False, times=32, chosen=1, q_func=None, origin=None):
        self.noise_ratio = self.max_noise_ratio
        old_state = state
        # q1, q2 = q_func(state, origin)
        # q_origin = 0.9 * torch.min(q1, q2) + 0.1 * torch.max(q1, q2)
        raw_batch_size = state.shape[0]
        state = state.repeat(times, 1)
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape)
        action.clamp_(-1., 1.)
        q1, q2 = q_func(state, action)
        q = torch.min(q1, q2)
        action = action.view(times, raw_batch_size, -1).transpose(0, 1)
        q = q.view(times, raw_batch_size, -1).transpose(0, 1)
        mean = q.mean()
        std = q.std()
        v = q.mean(dim=1, keepdim=True)

        # q_prox = torch.linalg.norm(action - origin.view(raw_batch_size, 1, -1), dim=-1, keepdim=True)
        q_prox = q #- 0.0 / math.sqrt(self.action_dim) * q.std(dim=1, keepdim=True) * q_prox

        if chosen == 1:
            _, q_idx = torch.max(q_prox, dim=1, keepdim=True)
            action_idx = q_idx.repeat(1, 1, self.action_dim)
            q = q.gather(dim=1, index=q_idx).view(raw_batch_size, 1)
            # q = torch.where(q>q_origin, q, v.view(raw_batch_size, 1))

            return old_state, action.gather(dim=1, index=action_idx).view(raw_batch_size, -1), (q.view(raw_batch_size, 1), v), (mean, std)
        else:
            q, q_idx = torch.topk(q, k=chosen, dim=1)
            action_idx = q_idx.repeat(1, 1, self.action_dim)
            return old_state.repeat(chosen, 1).view(chosen, raw_batch_size, -1).transpose(0,1).contiguous().view(raw_batch_size*chosen, -1), action.gather(dim=1, index=action_idx).view(raw_batch_size*chosen, -1), (q.view(raw_batch_size*chosen, 1), v), (mean, std)


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss


    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, eval=False, q_func=None, normal=False):
        #return self.dipo_sample(state, eval)
        return self.sample(state, eval, q_func, normal)
    
    # def set_timesteps(self, num_inference_steps, device=None):
    #     """
    #     Emulates Diffusers Scheduler.set_timesteps:
    #     build a reversed list of timesteps to use in inference.
    #     """
    #     #self.num_inference_steps = num_inference_steps
    #     # evenly‐spaced indices from 0...n_timesteps-1, then reverse
    #     timesteps = torch.linspace(
    #         0,
    #         self.n_timesteps - 1,
    #         num_inference_steps,
    #         device=device or self.betas.device
    #     ).long()
    #     # reverse order for backward process
    #     self.timesteps = timesteps.flip(0)

    def add_noise(self, x0, noise, timesteps):
        """
        Emulate Scheduler.add_noise: forward‐diffuse clean x0 to xt.
        """
        return self.q_sample(x_start=x0, t=timesteps, noise=noise)
    
    def scale_model_input(self, sample, timestep=None):
        """
        Emulate Scheduler.scale_model_input: optionally rescale xt
        before feeding into the denoiser network.
        """
        return extract(self.sqrt_recip_alphas_cumprod, timestep, sample.shape) * sample
    
    def step(self, model_output, timestep, sample):
        """
        Emulate Scheduler.step: reverse‐diffuse xt→x_{t-1} given εθ.
        """
        # 1) predict x0
        x0_pred = self.predict_start_from_noise(sample, t=timestep, noise=model_output)
        # 2) compute posterior mean & log‐variance
        mean, _, log_var = self.q_posterior(x_start=x0_pred, x_t=sample, t=timestep)
        # 3) add noise for backward step
        noise = torch.randn_like(sample)
        mask = (timestep > 0).float().view(-1,*([1]*(sample.ndim-1)))
        return mean + mask * (0.5 * log_var).exp() * noise * self.noise_ratio
    
    # inside class Diffusion:

    def __call__(self, state, eval=False, q_func=None, normal=False):
        """
        Route all policy calls through forward().
        We no longer support scheduler(step) dispatch here,
        since DDiffPG only needs to run the learned policy.
        """
        return self.forward(state, eval=eval, q_func=q_func, normal=normal)
        


    





