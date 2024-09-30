import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from multiprocessing_env import SubprocVecEnv

num_envs = 8
env_name = "CartPole-v0"


def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk


plt.ion()
envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)  # 8 env

env = gym.make(env_name)  # a single env


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        # dist = Categorical(probs) 使用 torch.distributions.Categorical 类
        # 创建了一个基于 probs 的类别分布对象 dist。
        dist = Categorical(probs)
        return dist, value


def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def plot(frame_idx, rewards):
    plt.plot(rewards, 'b-')
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.pause(0.0001)


num_inputs = envs.observation_space.shape[0]
num_outputs = envs.action_space.n

# Hyper params:
hidden_size = 256
lr = 1e-3
num_steps = 5

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters())

# 开始训练
# 收集经验,计算损失,反向传播
max_frames = 20000
frame_idx = 0
test_rewards = []

state = envs.reset()

while frame_idx < max_frames:

    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0

    # rollout trajectory
    # 现在模型展开num_steps步的轨迹：模型会根据观测状态返回动作的分布、状态价值，然后
    # 根据动作分布采样动作，接着环境step一步进入到下一个状态，并返回reward。
    # rollout trajectory
    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)
        # action = dist.sample() 从类别分布 dist 中随机采样一个动作 action。
        # 这个采样是根据概率分布 probs 进行的。
        action = dist.sample()
        next_state, reward, done, _ = envs.step(action.cpu().numpy())
        # log_prob = dist.log_prob(action) 计算所选动作 action 在概率分布 dist 中的对数概率。
        # dist.log_prob(action) 返回所选动作的对数概率，表示执行该动作的概率的对数值。
        # log_prob 的作用是用于策略梯度方法中计算损失函数。在强化学习中，
        # 策略的对数概率用于衡量当前策略的好坏，并指导策略的改进。
        # 具体来说，log_prob 是策略梯度算法（如A2C或PPO）的核心部分，
        # 用于调整策略网络的参数，使其能更好地选择有利的动作。
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

        state = next_state
        frame_idx += 1
        # 程序每隔100帧会进行一次评估，评估的方式是运行2次test_env()并计算返回的
        # total_reward的均值，这里用VisualDL记录它，文章的最后会展示模型运行效果。
        if frame_idx % 100 == 0:
            test_rewards.append(np.mean([test_env() for _ in range(10)]))
            plot(frame_idx, test_rewards)


    # 程序会记录展开轨迹的动作对数似然概率log_probs、模型估计价值values、回报rewards等，
    # 并计算优势值advantage 。由于是多环境并行，可以用paddle.concat将这些值分别拼接起来，
    # 随后计算出演员网络的损失actor_loss、评论家网络的损失critic_loss，在最终loss中有一项
    # 是动作分布熵的均值，希望能增大网络的探索能力。
    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    # 用VisualDL记录训练的actor_loss、critic_loss以及合并后的loss。然后再反向传播，优化神
    # 经网络的参数，开始下一轮的训练循环。
    writer.add_scalar("actor_loss", value=actor_loss, step=frame_idx)
    writer.add_scalar("critic_loss", value=critic_loss, step=frame_idx)
    writer.add_scalar("loss", value=loss, step=frame_idx)
    ##动态学习率，每隔2000帧缩放一次
    if frame_idx % 2000 == 0:
        lr = 0.92 * lr
        optimizer.set_lr(lr)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# test_env(True)