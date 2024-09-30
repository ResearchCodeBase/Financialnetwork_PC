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











if __name__ == '__main__':
    mode = "train"

    year = "2022"
    country = "America"
    type = 2
    ratio = "train0.6_val0.15_test0.25"
    dataset = torch.load(f'foreign_dataset/{country}/{year}/{ratio}/processed/BankingNetwork.dataset')
    country_name = "America_2022"
    # 加载银行重要性数据
    normalized_importance_values, max_importance_value = load_bank_importance(f'../Foreigh/data/{country_name}/{country}_bank_assets.csv')
    # 加载银行重要性数据

    if mode == "train":
        print("开始训练")
        epochs = 7500
        data = dataset[type]
        graph_model = GraphGCN(in_channels=data.num_features, data=data)
        model_path = f'save_models/{country}/{year}/{ratio}/best_model.pth'
        graph_model.load_state_dict(torch.load(model_path))
        # 通过调用 eval() 方法来完成。这样做会关闭某些特定于训练的功能（如 dropout 层），
        # 从而确保模型在测试时的行为与训练时一致。
        # 182*13 data.x
        graph_model.eval()  # 设置模型为测试模式
        print(data.x.size())
        policy_net = PolicyNetwork(in_channels=17, hidden_dim=16, num_nodes=data.num_nodes)
        value_net = ValueNetwork(in_channels=17, hidden_dim=16)
        a2c = A2C(policy_net, value_net)
        train_a2c(a2c, graph_model, data, data.edge_index, data.edge_weight, adjust_standardized_value_nonlinear,
              epochs=epochs,   normalized_importance_values= normalized_importance_values)
