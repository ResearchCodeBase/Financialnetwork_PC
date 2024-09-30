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
        # dist = Categorical(probs) ʹ�� torch.distributions.Categorical ��
        # ������һ������ probs �����ֲ����� dist��
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

# ��ʼѵ��
# �ռ�����,������ʧ,���򴫲�
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
    # ����ģ��չ��num_steps���Ĺ켣��ģ�ͻ���ݹ۲�״̬���ض����ķֲ���״̬��ֵ��Ȼ��
    # ���ݶ����ֲ��������������Ż���stepһ�����뵽��һ��״̬��������reward��
    # rollout trajectory
    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)
        # action = dist.sample() �����ֲ� dist ���������һ������ action��
        # ��������Ǹ��ݸ��ʷֲ� probs ���еġ�
        action = dist.sample()
        next_state, reward, done, _ = envs.step(action.cpu().numpy())
        # log_prob = dist.log_prob(action) ������ѡ���� action �ڸ��ʷֲ� dist �еĶ������ʡ�
        # dist.log_prob(action) ������ѡ�����Ķ������ʣ���ʾִ�иö����ĸ��ʵĶ���ֵ��
        # log_prob �����������ڲ����ݶȷ����м�����ʧ��������ǿ��ѧϰ�У�
        # ���ԵĶ����������ں�����ǰ���Եĺû�����ָ�����ԵĸĽ���
        # ������˵��log_prob �ǲ����ݶ��㷨����A2C��PPO���ĺ��Ĳ��֣�
        # ���ڵ�����������Ĳ�����ʹ���ܸ��õ�ѡ�������Ķ�����
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

        state = next_state
        frame_idx += 1
        # ����ÿ��100֡�����һ�������������ķ�ʽ������2��test_env()�����㷵�ص�
        # total_reward�ľ�ֵ��������VisualDL��¼�������µ�����չʾģ������Ч����
        if frame_idx % 100 == 0:
            test_rewards.append(np.mean([test_env() for _ in range(10)]))
            plot(frame_idx, test_rewards)


    # ������¼չ���켣�Ķ���������Ȼ����log_probs��ģ�͹��Ƽ�ֵvalues���ر�rewards�ȣ�
    # ����������ֵadvantage �������Ƕ໷�����У�������paddle.concat����Щֵ�ֱ�ƴ��������
    # ���������Ա�������ʧactor_loss�����ۼ��������ʧcritic_loss��������loss����һ��
    # �Ƕ����ֲ��صľ�ֵ��ϣ�������������̽��������
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

    # ��VisualDL��¼ѵ����actor_loss��critic_loss�Լ��ϲ����loss��Ȼ���ٷ��򴫲����Ż���
    # ������Ĳ�������ʼ��һ�ֵ�ѵ��ѭ����
    writer.add_scalar("actor_loss", value=actor_loss, step=frame_idx)
    writer.add_scalar("critic_loss", value=critic_loss, step=frame_idx)
    writer.add_scalar("loss", value=loss, step=frame_idx)
    ##��̬ѧϰ�ʣ�ÿ��2000֡����һ��
    if frame_idx % 2000 == 0:
        lr = 0.92 * lr
        optimizer.set_lr(lr)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# test_env(True)