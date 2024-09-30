import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv
from GCN.model.norm import GraphNorm
# 构建GraphSage模型
class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels,data):
        super(GraphGCN, self).__init__()
        torch.manual_seed(12345)
        # conv1.lin_src.weight: torch.Size([out_channels, data.num_features]) 是W的T  权重矩阵是8行14列
        # conv1.lin.weight: torch.Size([8, 14])
        self.conv1 = GCNConv(data.num_features, 16)
        # conv2.lin_src.weight: torch.Size([2, 8]) 权重矩阵是2行8列
        self.conv2 = GCNConv(16, 2)

    #     def forward(self, g: dgl.data.DGLDataset, feats, edge_weight=None):
    #         h = self.norm_layers[0](g, feats)
    #         h = self.layers[0](g, h, edge_weight=edge_weight)
    #         for n, l in zip(self.norm_layers[1:], self.layers[1:]):
    #             h = n(g, h)
    #             h = F.dropout(h, p=self.dropout, training=self.training)
    #             h = l(g, h, edge_weight=edge_weight)
    #         return h
    # 输入x,标准话，图卷积，激活，再标准化，丢弃/图卷积，再标准化，丢弃，图卷积，
    # 先标准化，droupout，再图卷积
    # 正确的做法，图卷积，激活，丢弃

    # def forward(self, x, edge_index,edge_weight):
    #     # 第一层输入权重矩阵
    #     x = self.conv1(x, edge_index,edge_weight)
    #     # print('第一层网络结果',x.shape)
    #     x = x.relu()
    #     # print('relu激活函数后', x.shape)
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     # print('droput后', x.shape)
    #     x = self.conv2(x, edge_index,edge_weight)
    #     # print('第二层网络结果', x.shape)
    #     # 注意这里输出的是节点的特征，维度为[节点数,类别数]
    #     # return x
    #     # 每一行元素和为1
    #     # print('Log_softmanx后',F.log_softmax(x, dim=1).shape)
    #     # 分类结果归一化
    #     # 选log 不要选 softmax
    #     return x
        # return F.softmax(x,dim=1)
    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        node_embeddings = x  # 保存第一层输出作为节点特征向量
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return node_embeddings, x
    # 后续在风险这里可以添加,银行本身节点的重要性因子
    def compute_node_risks(self, x, edge_index, edge_weight):
        with torch.no_grad():
            node_embeddings,risk_probs = self.forward(x, edge_index, edge_weight)
            probabilities = F.softmax(risk_probs, dim=1)[:, 1]  # 应用Softmax并取分类为1的概率
        return node_embeddings,probabilities

    def compute_total_risk(self, x, edge_index, edge_weight):
        _, probabilities = self.compute_node_risks(x, edge_index, edge_weight)
        total_risk = probabilities.sum().item()
        return total_risk


#     接收图 g，节点特征 feats，以及边权重 edge_weight。
# 首先对输入特征进行图归一化处理：h = self.norm_layers[0](g, feats)。
# 然后通过第一个图卷积层处理：h = self.layers[0](g, h, edge_weight=edge_weight)。
# 对于每一层（除了第一层）：
# 先进行图归一化处理：h = n(g, h)。
# 然后应用 dropout 层：h = F.dropout(h, p=self.dropout, training=self.training)。
# 接着通过图卷积层处理：h = l(g, h, edge_weight=edge_weight)。
# 最后返回经过所有层处理后的特征 h。


# 定义策略网络
# 定义策略网络
# 定义策略网络
# 定义基于GCN的策略网络 (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_nodes):
        super(PolicyNetwork, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_node = nn.Linear(hidden_dim, num_nodes)  # 输出num_nodes，表示每个节点的选择概率
        self.fc_ratio = nn.Linear(hidden_dim, 1)  # 输出1，表示救援比例

    def forward(self, x, edge_index, edge_weight):
        x = self.gcn1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.gcn2(x, edge_index, edge_weight)
        x = x.relu()

        # 节点选择概率
        node_logits = self.fc_node(x)
        # F.softmax 是 PyTorch 中的一个函数，用于将输入的 logits 转换为概率分布。
        # Softmax 函数会将输入的每个元素进行指数运算，并归一化，使得输出的元素和为 1。[0.6590, 0.2424, 0.0986]
        node_selector = F.softmax(node_logits, dim=0).squeeze()

        # 救援比例
        rescue_ratio = torch.sigmoid(self.fc_ratio(x)).squeeze()

        return node_selector, rescue_ratio


# 定义基于GCN的价值网络 (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)  # 输出为1，表示当前状态的价值

    def forward(self, x, edge_index, edge_weight, action):
        x = self.gcn1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.gcn2(x, edge_index, edge_weight)
        x = x.relu()
        # 加入动作空间
        x = torch.cat([x, action], dim=1)
        value = self.fc(x).squeeze()
        return value


# 定义PPO算法类
class PPO:
    def __init__(self, graph_model, policy_net, value_net, lr=0.01, gamma=0.99, epsilon=0.2):
        self.graph_model = graph_model
        self.policy_net = policy_net
        self.value_net = value_net
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer_policy = Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = Adam(self.value_net.parameters(), lr=lr)


    def select_action(self, node_embeddings, risk_probs, remaining_budget, edge_index, edge_weight):
        # 状态空间由节点嵌入和风险概率构成
        state = torch.cat((node_embeddings, risk_probs), dim=1)  # 合并节点特征和风险概率
        # 策略网络是输出选择节点与比例
        node_selector, raw_rescue_ratio = self.policy_net(state, edge_index, edge_weight)
        # 选择风险最大的个
        selected_node = torch.argmax(node_selector).item()
        # 约束，确保
        rescue_ratio = min(raw_rescue_ratio.item(), remaining_budget)  # 确保救援比例不超过剩余预算
        return selected_node, rescue_ratio
    # 策略网络更新：
    # 使用PPO的两个目标函数（surr1和surr2），通过比值ratio和优势函数advantage来计算策略损失。
    # 取最小值并取负号，是为了进行梯度下降，从而最小化策略损失。
    # 更新策略网络参数，使得新策略与旧策略之间的变化不超过一个范围（由epsilon决定），从而保证策略的稳定性。

    # 价值网络更新：
    # 通过计算当前状态和动作的预测价值与实际奖励之间的均方误差来更新价值网络。
    # 最小化均方误差损失，调整价值网络的参数，使其能够更准确地预测状态-动作对的价值。

    def update(self, states, actions, rewards, values, old_log_probs, edge_index, edge_weight):
        # 加入td_target
        advantages, returns = compute_advantage(rewards, values, self.gamma)
        for state, action, reward, advantage, old_log_prob in zip(states, actions, rewards, advantages, old_log_probs):
            selected_node, rescue_ratio = action
            node_selector, _ = self.policy_net(state, edge_index, edge_weight)

            node_prob = node_selector[selected_node]
            ratio = torch.exp(torch.log(node_prob) - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            action_tensor = torch.tensor([selected_node, rescue_ratio], dtype=torch.float32)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            value_loss = F.mse_loss(self.value_net(state, edge_index, edge_weight, action_tensor).mean(), reward_tensor)
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

# 计算优势函数

# 计算优势函数

# 计算优势函数
# 计算优势函数
def compute_advantage(rewards, values, gamma):

    #优势函数采样
    advantages = []
    advantage = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        td_error = r + gamma * advantage - v
        advantage = td_error + gamma * advantage
        advantages.insert(0, advantage)
    return advantages

# 调整标准化值
def adjust_standardized_value(current_value, allocated_budget, scale=0.1):
    adjustment = torch.sigmoid(torch.tensor(allocated_budget * scale))
    new_value = current_value + adjustment
    return new_value

def adjust_standardized_value_linear(current_value, rescue_ratio, max_value=1.0):
    new_value = current_value + rescue_ratio * (max_value - current_value)
    return new_value

# 改进后的奖励函数
def compute_reward(total_risk_before, total_risk_after, rescue_ratio, cost_weight=0.1, efficiency_weight=1.0):
    risk_reduction = total_risk_before - total_risk_after
    cost_penalty = rescue_ratio * cost_weight
    reward = efficiency_weight * risk_reduction - cost_penalty
    return reward
# 使用PPO算法处理金融系统救助的连续动作空间问题，可以有效地为每个节点分配最优的救援预算比例，
# 最大限度地降低系统性风险。PPO算法通过限制每次策略更新的步长，
# 保证了策略的稳定性和效率，非常适合处理这种复杂的金融系统优化问题。

# 训练策略网络和价值网络
# 训练函数
# 训练函数
# 训练函数

# 训练函数
def train_ppo(ppo, graph_model, data, edge_index, edge_weight, adjust_func, epochs=100):
    rewards_history = []
    total_risk_history = []
    node_rescue_ratios = torch.zeros(data.num_nodes)

    for epoch in range(epochs):
        total_reward = 0
        epoch_data_x = data.x.clone()
        remaining_budget = 1.0

        states, actions, rewards, values, old_log_probs = [], [], [], [], []

        for step in range(data.num_nodes):
            if remaining_budget <= 0:
                break

            node_embeddings, risk_probs = graph_model(epoch_data_x, edge_index, edge_weight)
            selected_node, rescue_ratio = ppo.select_action(node_embeddings, risk_probs, remaining_budget, edge_index, edge_weight)

            total_risk_before = graph_model.compute_total_risk(epoch_data_x, edge_index, edge_weight)
            epoch_data_x[selected_node, 0] = adjust_func(epoch_data_x[selected_node, 0], rescue_ratio)

            total_risk_after = graph_model.compute_total_risk(epoch_data_x, edge_index, edge_weight)
            reward = compute_reward(total_risk_before, total_risk_after, rescue_ratio)
            total_reward += reward

            rewards.append(reward)
            state_action = torch.cat((torch.cat((node_embeddings, risk_probs), dim=1), torch.tensor([[selected_node, rescue_ratio]], dtype=torch.float32)), dim=1)
            values.append(ppo.value_net(state_action, edge_index, edge_weight).mean().item())
            states.append(torch.cat((node_embeddings, risk_probs), dim=1))
            actions.append((selected_node, rescue_ratio))

            node_rescue_ratios[selected_node] += rescue_ratio
            remaining_budget -= rescue_ratio

            node_selector, _ = ppo.policy_net(torch.cat((node_embeddings, risk_probs), dim=1), edge_index, edge_weight)
            old_log_prob = torch.log(node_selector[selected_node])
            old_log_probs.append(old_log_prob)

        rewards_history.append(total_reward)
        total_risk_history.append(total_risk_after)

        ppo.update(states, actions, rewards, values, old_log_probs, edge_index, edge_weight)
        print(f"Epoch {epoch + 1}/{epochs}, Total Reward: {total_reward}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history, label='Total Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(total_risk_history, label='Total Risk After Rescue')
    plt.xlabel('Epoch')
    plt.ylabel('Total Risk')
    plt.legend()

    plt.show()

    node_rescue_ratios /= node_rescue_ratios.sum()
    for i, ratio in enumerate(node_rescue_ratios):
        print(f"Node {i}: Rescue Ratio: {ratio:.4f}")

if __name__ == '__main__':
    mode = "train"

    year = "2022"
    country = "China"
    type = 2
    ratio = "train0.6_val0.15_test0.25"
    dataset = torch.load(f'foreign_dataset/{country}/{year}/{ratio}/processed/BankingNetwork.dataset')

    if mode == "train":
        print("开始训练")
        epochs = 200
        data = dataset[type]
        graph_model = GraphGCN(in_channels=data.num_features, data=data)
        model_path = f'save_models/{country}/{year}/{ratio}/best_model.pth'
        graph_model.load_state_dict(torch.load(model_path))
        policy_net = PolicyNetwork(in_channels=data.num_features, hidden_dim=16, num_nodes=data.num_nodes)
        value_net = ValueNetwork(in_channels=data.num_features, hidden_dim=16)
        ppo = PPO(policy_net, value_net)
        train_ppo(ppo, graph_model, data, data.edge_index, data.edge_weight, adjust_standardized_value_linear, epochs=epochs)