"""
Topology Analysis of Deep Neural Networks in Reinforcement Learning
Based on the paper by Naitzat, Zhitnikov, and Lim (2020) & extending to gimnasium environment with mujoco hopper.
https://arxiv.org/pdf/2004.06093
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # For headless environments
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset, DataLoader
import gymnasium as gym
import networkx as nx
from tqdm import tqdm
from collections import deque
from scipy.spatial.distance import pdist, squareform

from ripser import ripser
from persim import plot_diagrams

import wandb
import yaml

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42) if torch.cuda.is_available() else None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#--------------------------#
# PPO Agent                #
#--------------------------#

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize NN layers with orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    """PPO Agent with policy and value networks that store activations for topology analysis."""
    
    def __init__(self, envs, hidden_size=64):
        super().__init__()
        
        self.activation = nn.Tanh()
            
        # Actor network
        self.actor_input = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size))
        self.actor_hidden = layer_init(nn.Linear(hidden_size, hidden_size))
        self.actor_mean = layer_init(nn.Linear(hidden_size, np.prod(envs.single_action_space.shape)), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        
        # Critic network
        self.critic_input = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size))
        self.critic_hidden = layer_init(nn.Linear(hidden_size, hidden_size))
        self.critic_output = layer_init(nn.Linear(hidden_size, 1), std=1.0)
        
        # Store intermediate activations
        self.reset_activations()
        
    def reset_activations(self):
        """Reset stored activations."""
        self.actor_activations = []
        self.critic_activations = []
    
    def get_value(self, x):
        """Forward pass through critic network."""
        x1 = self.activation(self.critic_input(x))
        x2 = self.activation(self.critic_hidden(x1))
        value = self.critic_output(x2)
        
        # Store activations for topology analysis (only in eval mode)
        if not self.training:
            self.critic_activations = [
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy()
            ]
        
        return value
    
    def get_action_and_value(self, x, action=None):
        """Forward pass through actor network and optionally critic."""
        
        x1 = self.activation(self.actor_input(x))
        x2 = self.activation(self.actor_hidden(x1))
        action_mean = self.actor_mean(x2)
        
        # Store activations for topology analysis (only in eval mode)
        if not self.training:
            self.actor_activations = [
                x1.detach().cpu().numpy(),
                x2.detach().cpu().numpy()
            ]
        
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.get_value(x)
        
        return action, log_prob, entropy, value

#--------------------------#
# Environment              #
#--------------------------#
class ClipObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, min_val=-10, max_val=10):
        super().__init__(env)
        self.min_val = min_val
        self.max_val = max_val
        
        # Update the observation space to reflect clipping
        original_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.maximum(original_space.low, min_val),
            high=np.minimum(original_space.high, max_val),
            shape=original_space.shape,
            dtype=original_space.dtype
        )
        
    def observation(self, observation):
        return np.clip(observation, self.min_val, self.max_val)


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda x: x % 100 == 0
            )
        
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = ClipObservationWrapper(env, -10, 10)
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -100, 100))
        
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return thunk

#-------------------------------#
# Topology Analysis Helper Func #
#-------------------------------#

def compute_graph_geodesic_distance(X, k=15):
    """
    Construct a k-NN graph and compute geodesic distance matrix.
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Handle empty input
    if X.shape[0] == 0:
        return np.array([[]])
    
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min(k, X.shape[0]), algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Create adjacency matrix
    n_samples = X.shape[0]
    graph = np.zeros((n_samples, n_samples))
    
    # Fill adjacency
    for i in range(n_samples):
        for j_idx, j in enumerate(indices[i]):
            graph[i, j] = distances[i, j_idx]
            graph[j, i] = distances[i, j_idx]
    
    # Turn into NetworkX graph
    G = nx.from_numpy_array(graph)
    
    # Remove zero-weight edges (self-loops)
    for u, v, d in list(G.edges(data=True)):
        if d['weight'] == 0:
            G.remove_edge(u, v)
    
    # Compute shortest paths
    try:
        path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))
        
        # Convert to distance matrix
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if j in path_lengths[i]:
                    dist_matrix[i, j] = path_lengths[i][j]
                else:
                    dist_matrix[i, j] = float('inf')
    except:
        # Fallback if graph is not connected
        print("Warning: Graph not connected. Using pairwise distances instead.")
        dist_matrix = squareform(pdist(X))
    
    return dist_matrix

def compute_persistent_homology(X, k=15, max_dim=2):
    """
    Compute persistent homology from data points.
    """
    # Handle empty input
    if X.shape[0] == 0:
        return {'dgms': [np.array([]) for _ in range(max_dim+1)]}
    
    # Compute distance matrix
    dist_matrix = compute_graph_geodesic_distance(X, k)
    
    # Replace inf with a large finite value
    inf_mask = np.isinf(dist_matrix)
    if np.any(inf_mask):
        max_finite = np.max(dist_matrix[~inf_mask]) if np.any(~inf_mask) else 1.0
        dist_matrix[inf_mask] = max_finite * 2
    
    # Compute persistence diagrams
    results = ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)
    return results

def compute_betti_numbers(ph_results, threshold=2.5):
    """
    Count features that persist past threshold in each dimension.
    """
    betti_numbers = []
    
    for dim, diagram in enumerate(ph_results['dgms']):
        # Empty diagram case
        if diagram.size == 0:
            betti_numbers.append(0)
            continue
        
        # Count features that are born before threshold and die after it
        count = np.sum((diagram[:, 0] <= threshold) & (diagram[:, 1] > threshold))
        betti_numbers.append(count)
    
    return betti_numbers

def sample_trajectories(agent, env, num_samples=100):
    """Sample states and actions from environment trajectories."""
    states = []
    agent.reset_activations()
    agent.eval()
    
    # Collect states from trajectories
    obs, _ = env.reset()
    done = False
    
    while len(states) < num_samples:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get action without storing in training buffers
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()
            
            # Execute step
            next_obs, _, terminated, truncated, _ = env.step(action[0])
            done = terminated or truncated
            
            # Store observation
            states.append(obs)
            
            # Reset if done
            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs
    
    # Convert to numpy array and take only what we need
    states = np.array(states[:num_samples])
    return states

def analyze_topology(agent, env, num_samples=500, k=15, threshold=2.5):
    """
    Analyze the topology of hidden layers in the actor and critic networks.
    
    Args:
        agent: The PPO agent
        env: Environment to sample states from
        num_samples: Number of states to sample
        k: Number of nearest neighbors for graph construction
        threshold: Persistence threshold for Betti numbers
    
    Returns:
        Dictionary with topology metrics
    """
    agent.eval()
    states = sample_trajectories(agent, env, num_samples)
    
    # Skip if we couldn't get enough samples
    if len(states) < 10:
        print("Not enough samples collected. Skipping topology analysis.")
        return None
    
    # Convert to tensor and get activations
    states_tensor = torch.FloatTensor(states).to(device)
    
    # Get activations without affecting training
    with torch.no_grad():
        # Forward pass to get activations
        agent.reset_activations()
        _, _, _, _ = agent.get_action_and_value(states_tensor)
    
    # Compute topology metrics
    results = {
        'input': {
            'ph': compute_persistent_homology(states, k=k, max_dim=min(2, states.shape[1]-1)),
            'data': states
        },
        'actor': [],
        'critic': []
    }
    
    # Analyze actor network activations
    for i, act in enumerate(agent.actor_activations):
        ph_results = compute_persistent_homology(act, k=k, max_dim=min(2, act.shape[1]-1))
        betti = compute_betti_numbers(ph_results, threshold=threshold)
        results['actor'].append({
            'layer': i+1,
            'ph': ph_results,
            'betti': betti,
            'data': act
        })
    
    # Analyze critic network activations
    for i, act in enumerate(agent.critic_activations):
        ph_results = compute_persistent_homology(act, k=k, max_dim=min(2, act.shape[1]-1))
        betti = compute_betti_numbers(ph_results, threshold=threshold)
        results['critic'].append({
            'layer': i+1,
            'ph': ph_results,
            'betti': betti,
            'data': act
        })
    
    return results

def visualize_topology_results(results, step, save_dir='results'):
    """
    Visualize topology analysis results.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Skip if no results
    if results is None:
        return
    
    # Plot Betti numbers for actor
    actor_betti = [compute_betti_numbers(results['input']['ph'])]
    actor_betti.extend([r['betti'] for r in results['actor']])
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(actor_betti))
    
    for dim in range(min(3, len(actor_betti[0]))):
        values = [layer[dim] if dim < len(layer) else 0 for layer in actor_betti]
        plt.plot(values, marker='o', label=f'β{dim}')
    
    plt.title(f'Actor Network Topology at Step {step}')
    plt.xlabel('Layer')
    plt.ylabel('Betti Number')
    plt.xticks(x, ['input'] + [f'layer {i+1}' for i in range(len(actor_betti)-1)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{save_dir}/actor_topology_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Betti numbers for critic
    critic_betti = [compute_betti_numbers(results['input']['ph'])]
    critic_betti.extend([r['betti'] for r in results['critic']])
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(critic_betti))
    
    for dim in range(min(3, len(critic_betti[0]))):
        values = [layer[dim] if dim < len(layer) else 0 for layer in critic_betti]
        plt.plot(values, marker='o', label=f'β{dim}')
    
    plt.title(f'Critic Network Topology at Step {step}')
    plt.xlabel('Layer')
    plt.ylabel('Betti Number')
    plt.xticks(x, ['input'] + [f'layer {i+1}' for i in range(len(critic_betti)-1)])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{save_dir}/critic_topology_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot persistence diagrams for select layers
    for network, name in [(results['actor'], 'actor'), (results['critic'], 'critic')]:
        for layer_result in network:
            layer_idx = layer_result['layer']
            
            plt.figure(figsize=(8, 8))
            plot_diagrams(layer_result['ph']['dgms'], show=False)
            plt.title(f'{name.capitalize()} Layer {layer_idx} Persistence Diagram at Step {step}')
            plt.savefig(f'{save_dir}/{name}_persistence_diagram_layer{layer_idx}_{step}.png', dpi=300, bbox_inches='tight')
            plt.close()

def visualize_topology_over_time(topology_results, save_dir='results'):
    """
    Create plots showing how topology evolves over training.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if len(topology_results) < 2:
        print("Not enough topology data to visualize over time")
        return
    
    # Extract time steps and betti numbers for actor network
    steps = [step for step, result in topology_results if result is not None]
    
    # Skip if no valid results
    if len(steps) == 0:
        return
    
    # Plot actor betti numbers over time
    actor_betti_over_time = {0: [], 1: [], 2: []}
    
    for step, result in topology_results:
        if result is None:
            continue
        
        for layer_idx, layer_data in enumerate(result['actor']):
            for dim, betti in enumerate(layer_data['betti']):
                if dim not in actor_betti_over_time:
                    continue
                
                # Ensure lists are initialized
                while len(actor_betti_over_time[dim]) <= layer_idx:
                    actor_betti_over_time[dim].append([])
                
                # Add data point
                if len(actor_betti_over_time[dim][layer_idx]) < len(steps):
                    actor_betti_over_time[dim][layer_idx].append(betti)
    
    # Create plots for each dimension
    for dim in actor_betti_over_time:
        plt.figure(figsize=(12, 6))
        
        for layer_idx, betti_values in enumerate(actor_betti_over_time[dim]):
            if len(betti_values) == len(steps):
                plt.plot(steps, betti_values, marker='o', label=f'Layer {layer_idx+1}')
        
        plt.title(f'Actor β{dim} Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel(f'Betti Number (β{dim})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{save_dir}/actor_betti{dim}_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot critic betti numbers over time (similar to actor)
    critic_betti_over_time = {0: [], 1: [], 2: []}
    
    for step, result in topology_results:
        if result is None:
            continue
        
        for layer_idx, layer_data in enumerate(result['critic']):
            for dim, betti in enumerate(layer_data['betti']):
                if dim not in critic_betti_over_time:
                    continue
                
                # Ensure lists are initialized
                while len(critic_betti_over_time[dim]) <= layer_idx:
                    critic_betti_over_time[dim].append([])
                
                # Add data point
                if len(critic_betti_over_time[dim][layer_idx]) < len(steps):
                    critic_betti_over_time[dim][layer_idx].append(betti)
    
    # Create plots for each dimension
    for dim in critic_betti_over_time:
        plt.figure(figsize=(12, 6))
        
        for layer_idx, betti_values in enumerate(critic_betti_over_time[dim]):
            if len(betti_values) == len(steps):
                plt.plot(steps, betti_values, marker='o', label=f'Layer {layer_idx+1}')
        
        plt.title(f'Critic β{dim} Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel(f'Betti Number (β{dim})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{save_dir}/critic_betti{dim}_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot total homology (sum of Betti numbers) over time
    plt.figure(figsize=(12, 6))
    
    actor_total = []
    critic_total = []
    
    for step, result in topology_results:
        if result is None:
            continue
        
        # Calculate total Betti numbers for actor
        actor_sum = sum(sum(b['betti']) for b in result['actor'])
        actor_total.append(actor_sum)
        
        # Calculate total Betti numbers for critic
        critic_sum = sum(sum(b['betti']) for b in result['critic'])
        critic_total.append(critic_sum)
    
    plt.plot(steps, actor_total, marker='o', label='Actor')
    plt.plot(steps, critic_total, marker='s', label='Critic')
    
    plt.title('Total Homology Over Training')
    plt.xlabel('Training Steps')
    plt.ylabel('Total Betti Numbers')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{save_dir}/total_homology_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()

#--------------------------#
# PPO Main Training        #
#--------------------------#

def train_hopper_with_topology_analysis(
    env_id="Hopper-v4",
    seed=42,
    hidden_szie=64,
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    num_envs=4,
    num_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=32,
    update_epochs=10,
    norm_adv=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None,
    topology_analysis_interval=10,
    save_checkpoint_interval=10,
    capture_video=True,
):
    """
    Train PPO on Hopper with topology analysis.
    """
    run_name = f"hopper_ppo_topo_{seed}_{int(time.time())}"

    wandb.init(project="hopper_ppo_topology_analysis", name=run_name)
    wandb.config.update({
        "env_id": env_id,
        "seed": seed,
        "hidden_szie": hidden_szie,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "num_minibatches": num_minibatches,
        "update_epochs": update_epochs,
        "norm_adv": norm_adv,
        "clip_coef": clip_coef,
        "clip_vloss": clip_vloss,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "target_kl": target_kl,
        "topology_analysis_interval": topology_analysis_interval,
        "save_checkpoint_interval": save_checkpoint_interval,
        "capture_video": capture_video
    })
    
    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"videos/{run_name}", exist_ok=True)
    
    # eval env for analysis
    analysis_env = gym.make(env_id)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed + i, i, capture_video, run_name) for i in range(num_envs)]
    )
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    agent = PPOAgent(envs, hidden_size=hidden_szie).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_timesteps // num_steps // num_envs
    
    topology_results = []
    
    print(f"Starting training for {num_updates} updates...")
    for update in range(1, num_updates + 1):
        # Decay learning rate linearly
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = learning_rate * frac
        optimizer.param_groups[0]["lr"] = lrnow
        
        # Rollout phase
        agent.train()
        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            if "final_info" in info:
                for item in info["final_info"]:
                    if item is not None and "episode" in item:
                        wandb.log({"charts/episodic_return": item["episode"]["r"]}, step=global_step)
                        wandb.log({"charts/episodic_length": item["episode"]["l"]}, step=global_step)
                        break
        
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values
        
        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Optimize policy and value networks
        b_inds = np.arange(num_steps * num_envs)
        clipfracs = []
        
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            
            # Mini-batch training
            for start in range(0, num_steps * num_envs, num_minibatches):
                end = start + num_minibatches
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
                
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                clipfracs.append(clipfrac)
            
            if target_kl is not None:
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    if approx_kl > target_kl:
                        break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        wandb.log({
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(global_step / (time.time() - start_time)),
        }, step=global_step)
        
        # Topology analysis at specified intervals
        if update % topology_analysis_interval == 0:
            print(f"Performing topology analysis at update {update}...")
            agent.eval()  # Set to evaluation mode
            topo_results = analyze_topology(agent, analysis_env)
            topology_results.append((global_step, topo_results))
            
            # Visualize current topology
            visualize_topology_results(topo_results, global_step)
            
            # Track Betti numbers over time
            if topo_results is not None:
                for i, layer_result in enumerate(topo_results['actor']):
                    for dim, betti in enumerate(layer_result['betti']):
                        wandb.log(
                            {f"topology/actor_layer{i+1}_betti{dim}": betti},
                            step=global_step
                        )
                
                for i, layer_result in enumerate(topo_results['critic']):
                    for dim, betti in enumerate(layer_result['betti']):
                        wandb.log(
                            {f"topology/critic_layer{i+1}_betti{dim}": betti},
                            step=global_step
                        )

        if update % save_checkpoint_interval == 0:
            save_path = f"checkpoints/{run_name}_{update}.pt"
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update': update,
                'global_step': global_step,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    # Final topology visualization over training
    visualize_topology_over_time(topology_results, save_dir='results')
    envs.close()
    analysis_env.close()
    wandb.finish()


if __name__ == "__main__":
    with open("configs/configs_rl.yml", "r") as f:
        config = yaml.safe_load(f)

    train_hopper_with_topology_analysis(**config)
