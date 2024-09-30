# -*- coding: utf-8 -*-
import datetime
import os
import random
import time

import numpy as np
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions import Categorical, Normal
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv
from GCN.model.norm import GraphNorm
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 构建GraphGCN模型
# 构建GraphGCN模型
class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels, data):
        super(GraphGCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index, edge_weight):
        x = x.to(device)  # 将输入数据移动到GPU
        edge_index = edge_index.to(device)  # 将边索引移动到GPU
        edge_weight = edge_weight.to(device)  # 将边权重移动到GPU
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        node_embeddings = x  # 保存第一层输出作为节点特征向量
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return node_embeddings, x

    def compute_node_risks(self, x, edge_index, edge_weight):
        x = x.to(device)  # 将输入数据移动到GPU
        edge_index = edge_index.to(device)  # 将边索引移动到GPU
        edge_weight = edge_weight.to(device)  # 将边权重移动到GPU
        with torch.no_grad():
            node_embeddings, risk_probs = self.forward(x, edge_index, edge_weight)
            probabilities = F.softmax(risk_probs, dim=1)[:, 1]
        return node_embeddings, probabilities

    def compute_total_risk(self, x, edge_index, edge_weight):
        _, probabilities = self.compute_node_risks(x, edge_index, edge_weight)
        total_risk = (probabilities.mean().item()) * 100
        return total_risk

# 定义基于GCN的策略网络 (Actor)

class PolicyNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_nodes):
        super(PolicyNetwork, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_node = nn.Linear(hidden_dim, 1)  # 输出1，表示每个节点的选择概率
        self.fc_rescue_ratio = nn.Linear(hidden_dim, 1)  # 救援比例的输出

        self.rescue_scale = 0.01 # 初始救援比例的缩放系数


    # def init_weights(self):
    #     # 使用 Xavier 初始化方法
    #     nn.init.xavier_uniform_(self.fc_node.weight)
    #     nn.init.constant_(self.fc_node.bias, 0)
    #     nn.init.xavier_uniform_(self.fc_ratio_mu.weight)
    #     nn.init.constant_(self.fc_ratio_mu.bias, 0)
    #     nn.init.xavier_uniform_(self.fc_ratio_sigma.weight)
    #     nn.init.constant_(self.fc_ratio_sigma.bias, 0)

    def forward(self, x, edge_index, edge_weight):
        # x = self.gcn1(x, edge_index, edge_weight)
        # x = x.relu()
        # x = self.gcn2(x, edge_index, edge_weight)
        # x = x.relu()
        # # print(f"Shape of x before fc_node: {x.shape}")  # 添加打印语句
        #
        # # 线性变换节点特征，得到每个节点的选择概率（logits）
        # node_logits = self.fc_node(x)  # 输出形状为 (num_nodes,)
        # # print(f"Shape of node_logits: {node_logits.shape}")
        #
        # # 对logits进行softmax
        # # 操作，得到每个节点被选择的概率分布
        # # F.softmax(node_logits, dim=0) 将logits转化为概率分布，使得所有元素的和为1
        # node_selector = F.softmax(node_logits, dim=0).squeeze()  # 输出形状为 (num_nodes,)
        # # print(f"Shape of node_selector: {node_selector.shape}")
        #
        # # 直接输出救援比例，并使用sigmoid函数将其限制在[0, 1]范围内，然后乘以缩放系数
        # rescue_ratios = torch.sigmoid(self.fc_rescue_ratio(x)).squeeze() * self.rescue_scale
        # return node_selector, rescue_ratios
        x = x.to(device)  # 将输入数据移动到GPU
        edge_index = edge_index.to(device)  # 将边索引移动到GPU
        edge_weight = edge_weight.to(device)  # 将边权重移动到GPU
        x = self.gcn1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.gcn2(x, edge_index, edge_weight)
        x = x.relu()
        node_logits = self.fc_node(x)
        node_selector = F.softmax(node_logits, dim=0).squeeze()
        rescue_ratios = torch.sigmoid(self.fc_rescue_ratio(x)).squeeze() * self.rescue_scale
        return node_selector, rescue_ratios


# 定义基于GCN的价值网络 (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)  # 输出为1，表示当前状态的价值

    def forward(self, x, edge_index, edge_weight):
        x = x.to(device)  # 将输入数据移动到GPU
        edge_index = edge_index.to(device)  # 将边索引移动到GPU
        edge_weight = edge_weight.to(device)  # 将边权重移动到GPU
        x = self.gcn1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.gcn2(x, edge_index, edge_weight)
        x = x.relu()
        value = self.fc(x).mean()  # 对所有节点的价值进行平均
        return value

# 定义A2C算法类
class A2C:
    def __init__(self, policy_net, value_net, lr=0.001, gamma=0.92, epsilon=0.3, epsilon_decay=0.99, min_epsilon=0.01):
        self.policy_net = policy_net.to(device)  # 将策略网络移动到GPU
        self.value_net = value_net.to(device)  # 将价值网络移动到GPU
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.optimizer_policy = Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = Adam(self.value_net.parameters(), lr=lr)
        self.scheduler_policy = StepLR(self.optimizer_policy, step_size=2000, gamma=0.92)
        self.scheduler_value = StepLR(self.optimizer_value, step_size=2000, gamma=0.92)

    def select_action(self, state, remaining_budget, edge_index, edge_weight):
        state = state.to(device)  # 将状态移动到GPU
        edge_index = edge_index.to(device)  # 将边索引移动到GPU
        edge_weight = edge_weight.to(device)  # 将边权重移动到GPU
        node_selector, rescue_ratios = self.policy_net(state, edge_index, edge_weight)
        node_selector = F.softmax(node_selector, dim=0)

        if torch.rand(1).item() < self.epsilon:
            selected_node = torch.randint(0, node_selector.size(0), (1,)).item()
        else:
            selected_node = Categorical(node_selector).sample().item()

        selected_node = torch.clamp(torch.tensor(selected_node), 0, node_selector.size(0) - 1).item()
        log_prob_node = torch.log(node_selector[selected_node] + 1e-10)

        rescue_ratio = rescue_ratios[selected_node].item()
        rescue_ratio = min(rescue_ratio, remaining_budget)
        log_prob_ratio = torch.log(rescue_ratios[selected_node] + 1e-10)

        log_prob = log_prob_node + log_prob_ratio

        return selected_node, rescue_ratio, log_prob

    def test_select_action(self, state, remaining_budget, edge_index, edge_weight):
        state = state.to(device)  # 将状态移动到GPU
        edge_index = edge_index.to(device)  # 将边索引移动到GPU
        edge_weight = edge_weight.to(device)  # 将边权重移动到GPU
        node_selector, rescue_ratios = self.policy_net(state, edge_index, edge_weight)

        selected_node = Categorical(node_selector).sample().item()

        selected_node = torch.clamp(torch.tensor(selected_node), 0, node_selector.size(0) - 1).item()
        log_prob_node = torch.log(node_selector[selected_node] + 1e-10)

        rescue_ratio = rescue_ratios[selected_node].item()
        rescue_ratio = min(rescue_ratio, remaining_budget)
        log_prob_ratio = torch.log(rescue_ratios[selected_node] + 1e-10)

        log_prob = log_prob_node + log_prob_ratio

        return selected_node, rescue_ratio, log_prob

    def update(self, states, actions, rewards, values, log_probs, next_value, edge_index, edge_weight, epoch):
        returns = compute_returns(next_value, rewards, self.gamma)
        log_probs = torch.stack(log_probs).to(device)  # 将log_probs移动到GPU
        values = torch.stack(values).to(device)  # 将values移动到GPU
        returns = torch.stack(returns).detach().to(device)  # 将returns移动到GPU
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        self.optimizer_policy.zero_grad()
        actor_loss.backward(retain_graph=True)

        # 梯度剪裁 可删除
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer_policy.step()
        self.scheduler_policy.step()

        self.optimizer_value.zero_grad()
        critic_loss.backward()

        # 梯度剪裁
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)

        self.optimizer_value.step()
        self.scheduler_value.step()

        # 打印当前学习率
        current_policy_lr = self.optimizer_policy.param_groups[0]['lr']
        current_value_lr = self.optimizer_value.param_groups[0]['lr']
        print(f'学习率: Policy LR: {current_policy_lr}, Value LR: {current_value_lr}')

        # 每2000幕更新学习率
        if (epoch + 1) % 2000 == 0:
            # 衰减 epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            print(f"Updated epsilon: {self.epsilon}")

#  引入外部资产作为银行的重要性
def load_bank_importance(file_path):
    df = pd.read_csv(file_path)
    importance_values = df['外部资产'].values
    min_value = importance_values.min()
    max_value = importance_values.max()


    # 将外部资产标准化到0~1之间
    normalized_importance_values = (importance_values - min_value) / (max_value - min_value)

    return normalized_importance_values, max_value

# 计算返回函数
def compute_returns(next_value, rewards, gamma):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R
        returns.insert(0, R)
    return returns


# 注入设计
def adjust_standardized_value_nonlinear(current_value, rescue_ratio, max_value=1, nonlinearity='log'):
    rescue_ratio = torch.tensor(rescue_ratio)  # 确保 rescue_ratio 是 Tensor 类型
    if nonlinearity == 'exp':
        new_value = current_value + (max_value - current_value) * (1 - torch.exp(-100 * rescue_ratio))
    elif nonlinearity == 'log':
        new_value = current_value + (max_value - current_value) * torch.log(1 + 100 * rescue_ratio)
    elif nonlinearity == 'sigmoid':
        new_value = current_value + (max_value - current_value) * torch.sigmoid(10 * rescue_ratio)
    elif nonlinearity == 'tanh':
        new_value = current_value + (max_value - current_value) * torch.tanh(5 * rescue_ratio)
    else:
        raise ValueError("Unsupported nonlinearity type")
    #
    # print('策略网络输出', rescue_ratio)
    # print('注入之后实际的差值', new_value - current_value)
    return new_value
#奖励函数设计
def compute_reward(total_risk_before, total_risk_after, node_risk_before, node_risk_after, rescue_ratio,
                   selected_node_assetvalue, normalized_importance_values,  system_weight=0.6, node_weight=0.4, scale_factor=10,
                   min_reward=1,
                   all_nodes_risk_probs=None):
    # 系统风险减少
    system_risk_reduction = total_risk_before - total_risk_after
    # print('系统风险_before', total_risk_before)
    # print('系统风险_after', total_risk_after)
    # print('======= 系统风险减少量', system_risk_reduction)
    normalized_system_risk_reduction = system_risk_reduction / total_risk_before if total_risk_before != 0 else 0

    # 节点风险减少
    node_risk_reduction = node_risk_before - node_risk_after
    # print('node_risk_before', node_risk_before)
    # print('node_risk_after', node_risk_after)
    # print('======= 节点风险减少量', node_risk_reduction)
    normalized_node_risk_reduction = node_risk_reduction / node_risk_before if node_risk_before != 0 else 0

    # 计算系统风险减少的效率
    system_efficiency = normalized_system_risk_reduction / (rescue_ratio + 1e-8)

    # 计算节点风险减少的效率
    node_efficiency = normalized_node_risk_reduction / (rescue_ratio + 1e-8)

    # # 节点选择概率权重
    # if node_risk_before > 0.75 and importance_weight > 0.75:
    #     selection_weight = 10  # 高风险高重要性
    # elif node_risk_before > 0.5 and importance_weight > 0.5:
    #     selection_weight = 5  # 中高风险和中高重要性
    # elif node_risk_before > 0.25 and importance_weight > 0.25:
    #     selection_weight = 2  # 中风险和中重要性
    # elif node_risk_before < 0.2 and importance_weight < 0.2:
    #     selection_weight = 0.5  # 低风险低重要性，惩罚
    # else:
    #     selection_weight = 1.0  # 其他情况

    # 节点选择概率权重
    # 计算四分位
    quartiles = np.percentile(normalized_importance_values, [25, 50, 75])
    # selected_node_assetvalue是选中的节点的标准化的后价值 normalized_importance_values是所有节点的标准化后的值
    # 节点选择概率权重
    # 节点选择概率权重
    # 节点选择概率权重
    if selected_node_assetvalue > quartiles[2]:  # 高于75%，高重要性
        if node_risk_before > 0.75:
            selection_weight = 10  # 高重要性高风险
        elif node_risk_before > 0.5:
            selection_weight = 5  # 高重要性中高风险
        elif node_risk_before > 0.25:
            selection_weight = 2  # 高重要性中风险
        else:
            selection_weight = 1  # 高重要性低风险
    elif selected_node_assetvalue > quartiles[1]:  # 50%到75%，中高重要性
        if node_risk_before > 0.75:
            selection_weight = 5  # 中高重要性高风险
        elif node_risk_before > 0.5:
            selection_weight = 2  # 中高重要性中高风险
        elif node_risk_before > 0.25:
            selection_weight = 1  # 中高重要性中风险
        else:
            selection_weight = 0.5  # 中高重要性低风险
    elif selected_node_assetvalue > quartiles[0]:  # 25%到50%，中重要性
        if node_risk_before > 0.75:
            selection_weight = 5  # 中重要性高风险
        elif node_risk_before > 0.5:
            selection_weight = 2  # 中重要性中高风险
        elif node_risk_before > 0.25:
            selection_weight = 0.5  # 中重要性中风险
        else:
            selection_weight = -1  # 中重要性低风险，负奖励
    else:  # 低于25%，低重要性
        if node_risk_before > 0.75:
            selection_weight = 2  # 低重要性高风险
        elif node_risk_before > 0.5:
            selection_weight = 1  # 低重要性中高风险
        elif node_risk_before > 0.25:
            selection_weight = 0.5  # 低重要性中风险
        else:
            selection_weight = -2  # 低重要性低风险，负奖励
    # 加权综合奖励
    reward = (system_weight * system_efficiency + node_weight * node_efficiency) * selection_weight
    # 对奖励进行缩放和放大



    # 确保奖励在合理范围内
    if system_risk_reduction < 0 or node_risk_reduction < 0:
        reward = -1  # 如果风险增加，奖励为负


    # 计算当前有多少节点风险降到了目标范围内
    nodes_below_threshold = sum(all_nodes_risk_probs < 0.4).item() if all_nodes_risk_probs is not None else 0

    # print('%%%%%%%%%%%%%%%%%%%%%%%%',all_nodes_risk_probs)
    total_nodes = len(all_nodes_risk_probs) if all_nodes_risk_probs is not None else 1
    proportion_below_threshold = nodes_below_threshold / total_nodes
    # print( 'proportion_below_threshold降低到0.4的整体概率，',proportion_below_threshold)

    # 阶段性奖励：当达到一定比例的节点风险降低到目标范围时，给予奖励,根据整体的风险给予奖励
    stage_rewards = [0.25, 0.5, 0.75, 1.0]  # 可以根据需要调整
    for stage in stage_rewards:
        if proportion_below_threshold >= stage:
            reward += stage * 10  # 可以根据需要调整阶段性奖励的值

    # # 最终奖励：所有节点风险都小于0.5时，给予大额奖励
    # if proportion_below_threshold == 1.0:
    #     reward += 5000  # 可以根据需要调整最终奖励的值
    # print('#################reward',reward)
    return reward

# 环境反馈函数
def environment_step(graph_model, epoch_data_x, selected_node, rescue_ratio, adjust_func, edge_index, edge_weight,
                     normalized_importance_values):
    # 计算采取动作前的系统风险和节点风险
    total_risk_before = graph_model.compute_total_risk(epoch_data_x.float().to(device), edge_index.to(device),
                                                       edge_weight.to(device))
    node_risk_before = \
    graph_model.compute_node_risks(epoch_data_x.float().to(device), edge_index.to(device), edge_weight.to(device))[1][
        selected_node]

    # 采取动作，注入救援资金
    updated_epoch_data_x = epoch_data_x.clone().to(device)  # 确保数据在GPU上
    updated_epoch_data_x[selected_node, 0] = adjust_func(epoch_data_x[selected_node, 0], rescue_ratio)

    # 计算采取动作后的系统风险和节点风险
    total_risk_after = graph_model.compute_total_risk(updated_epoch_data_x.float().to(device), edge_index.to(device),
                                                      edge_weight.to(device))
    node_risk_after = graph_model.compute_node_risks(updated_epoch_data_x.float().to(device), edge_index.to(device),
                                                     edge_weight.to(device))[1][selected_node]
    all_nodes_risk_probs = \
    graph_model.compute_node_risks(updated_epoch_data_x.float().to(device), edge_index.to(device),
                                   edge_weight.to(device))[1]

    selected_node_assetvalue = normalized_importance_values[selected_node]
    reward = compute_reward(total_risk_before, total_risk_after, node_risk_before, node_risk_after, rescue_ratio,
                            selected_node_assetvalue, normalized_importance_values,
                            all_nodes_risk_probs=all_nodes_risk_probs)

    node_embeddings, risk_probs = graph_model(updated_epoch_data_x.float().to(device), edge_index.to(device),
                                              edge_weight.to(device))
    risk_probs = F.softmax(risk_probs, dim=1)[:, 1].unsqueeze(1).to(device)
    next_state = torch.cat((node_embeddings, risk_probs), dim=1).to(device)

    return next_state, reward, updated_epoch_data_x
# 训练函数
def train_a2c(a2c, graph_model, data, edge_index, edge_weight, adjust_func, epochs=500, normalized_importance_values=None):
    rewards_history = []
    total_risk_history = []
    test_rewards_history = []
    test_risk_history = []
    best_reward = float('-inf')
    best_risk_probs = None
    best_rescue_ratios = None

    initial_node_risk_probs = graph_model.compute_node_risks(data.x.float().to(device), edge_index.to(device), edge_weight.to(device))[1]
    test_count = 1

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = f'training_results_{current_time}'
    os.makedirs(dir_name, exist_ok=True)

    filename = os.path.join(dir_name, f'test_rewards_and_risks.csv')
    risk_filename = os.path.join(dir_name, f'initial_and_best_risk_probs.csv')

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Epoch', 'Average Test Reward', 'Average Test Risk'])

    with open(risk_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Node', 'Initial Risk Probability', 'Best Risk Probability', 'Rescue Ratio'])

    for epoch in range(epochs):
        epoch_start_time = time.time()  # 记录开始时间

        total_reward = 0
        epoch_data_x = data.x.clone().to(device)  # 确保数据在GPU上
        remaining_budget = 1.0
        states, actions, rewards, values, log_probs = [], [], [], [], []
        count = 0
        node_rescue_ratios = torch.zeros(data.num_nodes).to(device)

        node_embeddings, risk_probs = graph_model(epoch_data_x.float().to(device), edge_index.to(device),
                                                  edge_weight.to(device))
        risk_probs = F.softmax(risk_probs, dim=1)[:, 1].unsqueeze(1).to(device)
        state = torch.cat((node_embeddings, risk_probs), dim=1).to(device)

        for step in range(data.num_nodes):
            if remaining_budget <= 0:
                break
            count = count + 1
            selected_node, rescue_ratio, log_prob = a2c.select_action(state, remaining_budget, edge_index, edge_weight)
            selected_node = selected_node.item() if isinstance(selected_node, torch.Tensor) else selected_node
            rescue_ratio = rescue_ratio.item() if isinstance(rescue_ratio, torch.Tensor) else rescue_ratio

            next_state, reward, update_epoch_data_x = environment_step(
                graph_model, epoch_data_x, selected_node, rescue_ratio, adjust_func, edge_index, edge_weight,
                normalized_importance_values)
            rewards.append(torch.tensor([reward]).to(device))  # 确保是张量并移动到GPU
            values.append(a2c.value_net(state, edge_index, edge_weight).unsqueeze(0).to(device))  # 确保是张量并移动到GPU
            states.append(state)

            log_probs.append(log_prob)

            node_rescue_ratios[selected_node] += rescue_ratio
            remaining_budget -= rescue_ratio
            epoch_data_x = update_epoch_data_x
            state = next_state
            total_reward += reward

        next_value = a2c.value_net(next_state, edge_index, edge_weight).unsqueeze(0).to(device)  # 确保是张量并移动到GPU

        a2c.update(states, actions, rewards, values, log_probs, next_value, edge_index, edge_weight, epoch)

        rewards_history.append(total_reward)
        if total_reward > best_reward:
            print('*****************hahahahahahahestrewar', best_reward)
            best_reward = total_reward
            best_risk_probs = graph_model.compute_node_risks(epoch_data_x.float().to(device), edge_index.to(device),
                                                             edge_weight.to(device))[1]
            best_rescue_ratios = node_rescue_ratios.clone()

            with open(risk_filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for i in range(len(initial_node_risk_probs)):
                    best_risk = best_risk_probs[i].item() if best_risk_probs is not None else 'N/A'
                    rescue_ratio = best_rescue_ratios[i].item() if best_rescue_ratios is not None else 'N/A'
                    csvwriter.writerow([i, initial_node_risk_probs[i].item(), best_risk, rescue_ratio])

        if (epoch + 1) % 50 == 0:
            average_test_reward, average_test_risk = test_policy(a2c, graph_model, data, edge_index, edge_weight,
                                                                 adjust_func,
                                                                 normalized_importance_values, dir_name, test_count)
            test_rewards_history.append(average_test_reward)
            test_risk_history.append(average_test_risk)

            with open(filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([(epoch + 1), average_test_reward, average_test_risk])

            print(f"Test after Epoch {epoch + 1}: Average Test Reward: {average_test_reward}")
            test_count = test_count + 1


        epoch_end_time = time.time()  # 记录结束时间
        epoch_duration = epoch_end_time - epoch_start_time  # 计算所花费的时间
        print(
            f"Epoch {epoch + 1}/{epochs}, Total Reward: {total_reward}, Step: {count}, Duration: {epoch_duration:.2f} seconds")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, label='Total Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(test_rewards_history, label='Average Test Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Average Test Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_risk_history, label='Average Test Risk')
    plt.xlabel('Epoch')
    plt.ylabel('Average Test Risk')
    plt.legend()

    plt.show()
    for i, ratio in enumerate(node_rescue_ratios):
        print(f"Node {i}: Rescue Ratio: {ratio:.4f}")

def test_policy(a2c, graph_model, data, edge_index, edge_weight, adjust_func, normalized_importance_values, dir_name,test_count,num_episodes=10):
    total_test_reward = 0
    total_test_risk = 0
    risk_changes = []
    node_rescue_sums = torch.zeros(data.num_nodes).to(device)  # 将张量移动到GPU
    node_risk_sums = torch.zeros(data.num_nodes).to(device)  # 将张量移动到GPU

    for _ in range(num_episodes):
        epoch_data_x = data.x.clone().to(device)  # 将数据移动到GPU
        remaining_budget = 1.0
        node_rescue_ratios = torch.zeros(data.num_nodes).to(device)  # 将张量移动到GPU

        node_embeddings, risk_probs = graph_model(epoch_data_x.float().to(device), edge_index.to(device),
                                                  edge_weight.to(device))
        risk_probs = F.softmax(risk_probs, dim=1)[:, 1].unsqueeze(1).to(device)
        state = torch.cat((node_embeddings, risk_probs), dim=1).to(device)

        test_reward = 0

        for step in range(data.num_nodes):
            if remaining_budget <= 0:
                break
            selected_node, rescue_ratio, _ = a2c.test_select_action(state, remaining_budget, edge_index, edge_weight)
            selected_node = selected_node.item() if isinstance(selected_node, torch.Tensor) else selected_node
            rescue_ratio = rescue_ratio.item() if isinstance(rescue_ratio, torch.Tensor) else rescue_ratio

            next_state, reward, update_epoch_data_x = environment_step(
                graph_model, epoch_data_x, selected_node, rescue_ratio, adjust_func, edge_index, edge_weight,
                normalized_importance_values)
            test_reward += reward

            node_rescue_ratios[selected_node] += rescue_ratio
            remaining_budget -= rescue_ratio
            epoch_data_x = update_epoch_data_x
            state = next_state

        test_risk = graph_model.compute_total_risk(epoch_data_x.float().to(device), edge_index.to(device),
                                                   edge_weight.to(device))
        total_test_risk += test_risk
        total_test_reward += test_reward
        node_rescue_sums += node_rescue_ratios
        node_risk_sums += \
        graph_model.compute_node_risks(epoch_data_x.float().to(device), edge_index.to(device), edge_weight.to(device))[
            1]

    average_test_reward = total_test_reward / num_episodes
    average_test_risk = total_test_risk / num_episodes
    average_node_rescues = node_rescue_sums / num_episodes
    average_node_risks = node_risk_sums / num_episodes

    average_node_rescues = average_node_rescues.tolist()
    average_node_risks = average_node_risks.tolist()

    file_path = os.path.join(dir_name, f'node_averages_{test_count}.csv')

    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Node', 'Average Rescue Ratio', 'Average Risk Probability'])
        for i in range(len(average_node_rescues)):
            csvwriter.writerow([i, average_node_rescues[i], average_node_risks[i]])

    return average_test_reward, average_test_risk


if __name__ == '__main__':
    mode = "train"

    year = "2022"
    country = "China"
    type = 2
    ratio = "train0.6_val0.15_test0.25"
    dataset = torch.load(f'foreign_dataset/{country}/{year}/{ratio}/processed/BankingNetwork.dataset')
    # 加载银行重要性数据
    normalized_importance_values, max_importance_value = load_bank_importance(
        f'../Foreigh/China/2022/bank_importance.csv')
    # 加载银行重要性数据

    if mode == "train":
        print("开始训练")
        epochs = 30000
        data = dataset[type]
        graph_model = GraphGCN(in_channels=data.num_features, data=data)
        model_path = f'save_models/{country}/{year}/{ratio}/best_model.pth'
        graph_model.load_state_dict(torch.load(model_path))
        # 通过调用 eval() 方法来完成。这样做会关闭某些特定于训练的功能（如 dropout 层），
        # 从而确保模型在测试时的行为与训练时一致。
        # 182*13 data.x
        graph_model = graph_model.to(device)
        graph_model.eval()  # 设置模型为测试模式
        print(data.x.size())
        policy_net = PolicyNetwork(in_channels=17, hidden_dim=16, num_nodes=data.num_nodes)
        value_net = ValueNetwork(in_channels=17, hidden_dim=16)
        a2c = A2C(policy_net, value_net)
        train_a2c(a2c, graph_model, data, data.edge_index, data.edge_weight, adjust_standardized_value_nonlinear,
              epochs=epochs,   normalized_importance_values= normalized_importance_values)
