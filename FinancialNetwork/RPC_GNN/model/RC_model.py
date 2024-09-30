
# -*- coding: utf-8 -*-
from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn.functional as F

from torch.distributions import Categorical, Normal
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv

# 构建GraphGCN模型
class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels, data):
        super(GraphGCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        node_embeddings = x  # 保存第一层输出作为节点特征向量
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return node_embeddings,  x

    def compute_node_risks(self, x, edge_index, edge_weight):

        with torch.no_grad():
            node_embeddings, risk_probs = self.forward(x, edge_index, edge_weight)
            probabilities = F.softmax(risk_probs, dim=1)[:, 1]
        return node_embeddings, probabilities


    def compute_total_risk(self, x, edge_index, edge_weight):
        _, probabilities = self.compute_node_risks(x, edge_index, edge_weight)
        # 将风险概率标准化到0~100之间
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

        self.rescue_scale = 0.08 # 初始救援比例的缩放系数


    # def init_weights(self):
    #     # 使用 Xavier 初始化方法
    #     nn.init.xavier_uniform_(self.fc_node.weight)
    #     nn.init.constant_(self.fc_node.bias, 0)
    #     nn.init.xavier_uniform_(self.fc_ratio_mu.weight)
    #     nn.init.constant_(self.fc_ratio_mu.bias, 0)
    #     nn.init.xavier_uniform_(self.fc_ratio_sigma.weight)
    #     nn.init.constant_(self.fc_ratio_sigma.bias, 0)

    def forward(self, x, edge_index, edge_weight):
        x = self.gcn1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.gcn2(x, edge_index, edge_weight)
        x = x.relu()
        # print(f"Shape of x before fc_node: {x.shape}")  # 添加打印语句

        # 线性变换节点特征，得到每个节点的选择概率（logits）
        node_logits = self.fc_node(x)  # 输出形状为 (num_nodes,)
        # print(f"Shape of node_logits: {node_logits.shape}")

        # 对logits进行softmax
        # 操作，得到每个节点被选择的概率分布
        # F.softmax(node_logits, dim=0) 将logits转化为概率分布，使得所有元素的和为1
        node_selector = F.softmax(node_logits, dim=0).squeeze()  # 输出形状为 (num_nodes,)
        # print(f"Shape of node_selector: {node_selector.shape}")

        # 直接输出救援比例，并使用sigmoid函数将其限制在[0, 1]范围内，然后乘以缩放系数
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
        x = self.gcn1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.gcn2(x, edge_index, edge_weight)
        x = x.relu()
        value = self.fc(x).mean()  # 对所有节点的价值进行平均
        # print('value价值网络输出', value)
        return value

# 定义A2C算法类
class A2C:
    def __init__(self, policy_net, value_net, lr=0.001, gamma=0.92, epsilon=0.3, epsilon_decay=0.99, min_epsilon=0.01):
        self.policy_net = policy_net
        self.value_net = value_net
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.optimizer_policy = Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = Adam(self.value_net.parameters(), lr=lr)
        self.scheduler_policy = StepLR(self.optimizer_policy, step_size=1000, gamma=0.92)
        self.scheduler_value = StepLR(self.optimizer_value, step_size=1000, gamma=0.92)

    def select_action(self, state, remaining_budget, edge_index, edge_weight):
        node_selector, rescue_ratios = self.policy_net(state, edge_index, edge_weight)
        # 使用 Softmax 将 node_selector 转换为概率分布
        node_selector = F.softmax(node_selector, dim=0)

        if torch.rand(1).item() < self.epsilon:
            selected_node = torch.randint(0, node_selector.size(0), (1,)).item()
        else:
            selected_node = Categorical(node_selector).sample().item()

        # 确保索引不超出范围
        selected_node = torch.clamp(torch.tensor(selected_node), 0, node_selector.size(0) - 1).item()
        # print('截取范围后动作索引', selected_node)

        # 计算所选节点的对数概率，用于计算策略的损失
        log_prob_node = torch.log(node_selector[selected_node]+ + 1e-10)

        # 直接使用策略网络输出的救援比例
        rescue_ratio = rescue_ratios[selected_node].item()
        rescue_ratio = min(rescue_ratio, remaining_budget)  # 确保救援比例不超过剩余预算
        # print('策略网络救援比例', rescue_ratio)
        log_prob_ratio = torch.log(rescue_ratios[selected_node]+ + 1e-10)

        # 总的对数概率是节点选择和救援比例的对数概率之和
        log_prob = log_prob_node + log_prob_ratio

        return selected_node, rescue_ratio, log_prob

    def test_select_action(self, state, remaining_budget, edge_index, edge_weight):
        node_selector, rescue_ratios = self.policy_net(state, edge_index, edge_weight)


        selected_node = Categorical(node_selector).sample().item()

        # 确保索引不超出范围
        selected_node = torch.clamp(torch.tensor(selected_node), 0, node_selector.size(0) - 1).item()
        # print('截取范围后动作索引', selected_node)

        # 计算所选节点的对数概率，用于计算策略的损失
        log_prob_node = torch.log(node_selector[selected_node]++ 1e-10)

        # 直接使用策略网络输出的救援比例
        rescue_ratio = rescue_ratios[selected_node].item()
        rescue_ratio = min(rescue_ratio, remaining_budget)  # 确保救援比例不超过剩余预算
        # print('策略网络救援比例', rescue_ratio)
        log_prob_ratio = torch.log(rescue_ratios[selected_node]+ 1e-10)

        # 总的对数概率是节点选择和救援比例的对数概率之和
        log_prob = log_prob_node + log_prob_ratio

        return selected_node, rescue_ratio, log_prob

    def update(self, states, actions, rewards, values, log_probs, next_value, edge_index, edge_weight, epoch):
        returns = compute_returns(next_value, rewards, self.gamma)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        returns = torch.stack(returns).detach()
        advantage = returns - values
        # print('advantage',advantage)
        # print('returns',returns)
        # print('values',values)

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
