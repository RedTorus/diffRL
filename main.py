import argparse
import copy

import numpy as np
import torch

from agent.qvpo import QVPO
from agent.dipo import DiPo
from agent.ddiffpg import DDiffPG
from agent.replay_memory import ReplayMemory, DiffusionMemory

from tensorboardX import SummaryWriter
import gym
import os
from logger import Logger
import datetime
import wandb
from argparse import Namespace
import yaml
import pdb
import ast
import d4rl

def readParser():
    parser = argparse.ArgumentParser(description='Diffusion Policy')
    parser.add_argument('--env_name', default="Hopper-v3",
                        help='Mujoco Gym environment (default: Hopper-v3)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')

    parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                        help='env timesteps (default: 1000000)') #1000 000

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--update_actor_target_every', type=int, default=1, metavar='N',
                        help='update actor target per iteration (default: 1)')

    parser.add_argument("--policy_type", type=str, default="Diffusion", metavar='S',
                        help="Diffusion, VAE or MLP")
    parser.add_argument("--beta_schedule", type=str, default="cosine", metavar='S',
                        help="linear, cosine or vp")
    parser.add_argument('--n_timesteps', type=int, default=20, metavar='N',
                        help='diffusion timesteps (default: 20)')
    parser.add_argument('--diffusion_lr', type=float, default=0.0001, metavar='G',
                        help='diffusion learning rate (default: 0.0001)')
    parser.add_argument('--critic_lr', type=float, default=0.0003, metavar='G',
                        help='critic learning rate (default: 0.0003)')
    parser.add_argument('--action_lr', type=float, default=0.03, metavar='G',
                        help='diffusion learning rate (default: 0.03)')
    parser.add_argument('--noise_ratio', type=float, default=1.0, metavar='G',
                        help='noise ratio in sample process (default: 1.0)')

    parser.add_argument('--action_gradient_steps', type=int, default=20, metavar='N',
                        help='action gradient steps (default: 20)')
    parser.add_argument('--ratio', type=float, default=0.1, metavar='G',
                        help='the ratio of action grad norm to action_dim (default: 0.1)')
    parser.add_argument('--ac_grad_norm', type=float, default=2.0, metavar='G',
                        help='actor and critic grad norm (default: 1.0)')

    parser.add_argument('--cuda', default='cuda:0',
                        help='run on CUDA (default: cuda:0)')

    parser.add_argument('--alpha_mean', type=float, default=0.001, metavar='G',
                        help='running mean update weight (default: 0.1)')

    parser.add_argument('--alpha_std', type=float, default=0.001, metavar='G',
                        help='running std update weight (default: 0.001)')

    parser.add_argument('--beta', type=float, default=1.0, metavar='G',
                        help='expQ weight (default: 1.0)')

    parser.add_argument('--weighted', action="store_true", help="weighted training")

    parser.add_argument('--aug', action="store_true", help="augmentation")

    parser.add_argument('--train_sample', type=int, default=64, metavar='N',
                        help='train_sample (default: 64)')

    parser.add_argument('--chosen', type=int, default=1, metavar='N', help="chosen actions (default:1)")

    parser.add_argument('--q_neg', type=float, default=0.0, metavar='G', help="q_neg (default: 0.0)")

    parser.add_argument('--behavior_sample', type=int, default=4, metavar='N', help="behavior_sample (default: 1)")
    parser.add_argument('--target_sample', type=int, default=4, metavar='N', help="target_sample (default: behavior sample)")

    parser.add_argument('--eval_sample', type=int, default=32, metavar='N', help="eval_sample (default: 512)")

    parser.add_argument('--deterministic', action="store_true", help="deterministic mode")

    parser.add_argument('--q_transform', type=str, default='qadv', metavar='S', help="q_transform (default: qrelu)")

    parser.add_argument('--gradient', action="store_true", help="aug gradient")

    parser.add_argument('--policy_freq', type=int, default=1, metavar='N', help="policy_freq (default: 1)")

    parser.add_argument('--cut', type=float, default=1.0, metavar='G', help="cut (default: 1.0)")
    parser.add_argument('--times', type=int, default=1, metavar='N', help="times (default: 1)")

    parser.add_argument('--epsilon', type=float, default=0.0, metavar='G', help="eps greedy (default: 0.0)")
    parser.add_argument('--entropy_alpha', type=float, default=0.02, metavar='G', help="entropy_alpha (default: 0.02)")
    parser.add_argument('--use_wandb', default=True, help="Enable wandb logging")

    parser.add_argument('--agent', type=str, default='qvpo', help="qvpo or dipo")
    parser.add_argument('--diffusion_mode' , type=str, default='ddpm', help="ddpm or ddim")
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--pretraining_steps', type=int, default=0)

    return parser.parse_args()


def evaluate(env, agent, steps):
    episodes = 10
    returns = np.zeros((episodes,), dtype=np.float32)

    for i in range(episodes):
        state = env.reset()
        episode_reward = 0.
        done = False
        while not done:
            action = agent.sample_action(state, eval=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        returns[i] = episode_reward

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    print('-' * 60)
    print(f'Num steps: {steps:<5}  '
          f'reward: {mean_return:<5.1f}  '
          f'std: {std_return:<5.1f}')
    print(returns)
    print('-' * 60)
    return mean_return

def generate_run_name(prefix='run'):
    """
    Generates a unique run name based on the current day, month, and time (hours and minutes).
    
    Parameters:
    - prefix (str): A string prefix to add before the date. Default is 'run'.
    
    Returns:
    - run_name (str): A unique run name formatted as '<prefix>_DDMM_HHMM'.
    """
    now = datetime.datetime.now()
    # Format: day (DD), month (MM), underscore, hour (HH) and minute (MM)
    run_name = f"{prefix}_{now.strftime('%d%m_%H%M')}"
    return run_name

def init_wandb(args, project_name="DiffRL", run_name=None):
    """
    Initializes Weights & Biases for experiment tracking.

    Parameters:
    - args: An argparse.Namespace object or a dictionary containing your experiment's configuration.
            This is usually the parsed arguments from your command-line interface.
    - project_name (str): The name of the wandb project to log the run under. Default is "qvpo".
    - run_name (str, optional): A custom name for this specific run (optional).

    Returns:
    - run: The wandb run object for further logging, if needed.
    """
    # Convert args to dictionary if necessary.
    config = args if isinstance(args, dict) else vars(args)
    Name = generate_run_name() if run_name is None else generate_run_name(run_name)
    # Initialize a new wandb run
    run = wandb.init(
        project=project_name,
        name=Name,
        config=config,
        reinit=True  # Allows running multiple runs in the same process if needed.
    )

    
    
    return run

def merge_yaml_into_args(args, yaml_path="agent/config.yaml"):
    """
    Given an argparse.Namespace `args` (already from parser.parse_args()),
    load `config.yaml`, and override any fields under `args:`.

    • If a YAML value is a string, we try ast.literal_eval() to convert
      numerics (e.g. '1e-4') into Python numbers.
    • If literal_eval fails (e.g. for 'vp'), we keep the original string.
    """
    # 1) Load the YAML config as a dict
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)                                  # :contentReference[oaicite:0]{index=0}

    # 2) Merge under the 'args' section
    for key, val in cfg.get("args", {}).items():
        if isinstance(val, str):
            try:
                # converts '1e-4'→0.0001, '200_000'→200000, '[1,2,3]'→[1,2,3], etc.
                val = ast.literal_eval(val)                      # :contentReference[oaicite:1]{index=1}
            except (ValueError, SyntaxError):
                # leave non‑literal strings like "vp" intact
                pass
        setattr(args, key, val)

    return args

def main(args=None, logger=None, id=None):

    device = torch.device(args.cuda)

    dir = "record"
    # dir = "test"
    log_dir = os.path.join(dir, f'{args.env_name}', f'policy_type={args.policy_type}', f'ratio={args.ratio}',
                           f'seed={args.seed}')
    writer = SummaryWriter(log_dir)

    if args.use_wandb:
        print("--------Using wandb for logging")
        init_wandb(args, run_name=args.agent+args.env_name+ args.diffusion_mode)
        wandb.define_metric("critic_loss", step_metric="step")
        wandb.define_metric("actor_loss",  step_metric="step")
        wandb.define_metric("reward",      step_metric="episode")
        wandb.define_metric("steps",       step_metric="episode")
    # Initial environment
    env = gym.make(args.env_name)
    #env = gym.wrappers.TimeLimit(env, max_episode_steps=300)
    if args.env_name in ['pen-human-v0', 'pen-cloned-v0', 'pen-expert-v0', 'relocate-human-v0', 'relocate-cloned-v0', 'relocate-expert-v0', 'hammer-human-v0', 'hammer-cloned-v0', 'hammer-expert-v0']:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=300)
        dataset = d4rl.qlearning_dataset(env) #env.get_dataset()
    else:
        dataset=None
    eval_env = copy.deepcopy((env))
    state_size = int(np.prod(env.observation_space.shape))
    action_size = int(np.prod(env.action_space.shape))
    print(action_size)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    eval_env.seed(args.seed)

    memory_size = 1e6
    num_steps = args.num_steps
    start_steps = args.start_steps
    eval_interval = 10000
    updates_per_step = 1
    batch_size = args.batch_size
    log_interval = 10
    pretraining_steps = args.pretraining_steps

    memory = ReplayMemory(state_size, action_size, memory_size, device)
    diffusion_memory = DiffusionMemory(state_size, action_size, memory_size, device)
    merge_yaml_into_args(args)
    #pdb.set_trace()
    cfg = Namespace(
        memory = memory,
        diffusion_memory = diffusion_memory,
        action_dim = action_size,
        state_dim = state_size,
        action_space = env.action_space,
        device = device,
        updates_per_step=updates_per_step,
        gamma=args.gamma,
        args=args,
    )

    #agent = QVPO(args, state_size, env.action_space, memory, diffusion_memory, device) if args.agent == 'qvpo' else \
    #    DiPo(args, state_size, env.action_space, memory, diffusion_memory, device)
    #agent = QVPO(cfg) #QVPO(cfg) #if args.agent == 'qvpo' else DiPo(cfg)
    #agent = DiPo(args, state_size, env.action_space, memory, diffusion_memory, device)
    #QVPO(args, state_size, env.action_space, memory, diffusion_memory, device)
    if args.agent == 'qvpo':
        agent = QVPO(cfg)
    elif args.agent == 'dipo':
        agent = DiPo(cfg)
    elif args.agent == 'ddiffpg':
        agent = DDiffPG(cfg) 
    #DDiffPG(cfg)
    steps = 0
    episodes = 0
    best_result = -float('inf')

    if dataset is not None:
        obs    = dataset["observations"]
        acts   = dataset["actions"]
        rews   = dataset["rewards"]
        nobs   = dataset["next_observations"]
        dones  = dataset["terminals"]


        for s, a, r, s2, d in zip(obs, acts, rews, nobs, dones):
            mask = 0.0 if d else cfg.args.gamma
            agent.append_memory(s, a, r, s2, mask)

        for _ in range(pretraining_steps):
            agent.train(log_callback=lambda metrics, step: wandb.log(metrics))
    

    while steps < num_steps:
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = env.reset()
        episodes += 1
        while not done:
            if start_steps > steps:
                action = env.action_space.sample()
            else:
                action = agent.sample_action(state, eval=False)
            next_state, reward, done, _ = env.step(action)

            mask = 0.0 if done else args.gamma

            steps += 1
            episode_steps += 1
            episode_reward += reward
            #print(f"steps: {steps}, reward: {reward}")
            agent.append_memory(state, action, reward, next_state, mask)

            if steps >= start_steps:
                #agent.train(steps, updates_per_step, batch_size=batch_size, log_writer=writer)
                #agent.train()
                if args.agent!='ddiffpg':
                    agent.train(log_callback=lambda metrics, step: wandb.log(metrics))
                else:
                    agent.train(steps, updates_per_step, log_callback=lambda metrics, step: wandb.log(metrics))

                if args.agent == 'qvpo':
                    agent.entropy_alpha = min(args.entropy_alpha, max(0.002, args.entropy_alpha-steps/num_steps*args.entropy_alpha))
            #print("train and entropy done")
            if steps % eval_interval == 0:
                tmp_result = evaluate(eval_env, agent, steps)
                if tmp_result > best_result:
                    best_result = tmp_result
                    agent.save_model(os.path.join('./results', prefix + '_' + name), id=id)
                # self.save_models()

            state = next_state

        # if episodes % log_interval == 0:
        #     writer.add_scalar('reward/train', episode_reward, steps)
        if args.use_wandb:
            wandb.log({
                "reward": episode_reward,
                "steps": episode_steps,
                "episode": episodes,
            })

        print(f'episode: {episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'reward: {episode_reward:<5.1f}'
                f'  steps: {steps:<5}  ')

        if logger is not None:
            for i in range(episode_steps):
                logger.add(epoch=steps-episode_steps+i, reward=episode_reward)


if __name__ == "__main__":
    args = readParser()
    if args.target_sample == -1:
        args.target_sample = args.behavior_sample


    ## settings
    prefix = 'qvpo'
    name = args.env_name
    keys = ("epoch", "reward")
    times = args.times
    id = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    logger = Logger(name=name, keys=keys, max_epochs=int(args.num_steps)+2100, times=times, config=args, path=os.path.join('./results', prefix + '_' + name), id=id)


    ## run
    for time in range(times):
        main(args, logger=logger, id=id+"_"+str(time))

    logger.save(os.path.join('./results', prefix + '_' + name), id=id)
