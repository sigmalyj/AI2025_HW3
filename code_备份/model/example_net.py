import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce

from env.base_env import BaseGame

class BaseNetConfig:
    def __init__(
        self, 
        num_channels:int = 256,
        dropout:float = 0.3,
        linear_hidden:list[int] = [256, 128],
    ):
        self.num_channels = num_channels
        self.linear_hidden = linear_hidden
        self.dropout = dropout
        
class MLPNet(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        input_dim = observation_size[0] * observation_size[1] if len(observation_size) == 2 else observation_size[0]
        self.layer1 = nn.Linear(input_dim, config.linear_hidden[0])
        self.layer2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])
        
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)
        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, x: torch.Tensor):
        #                                                         x: batch_size x board_x x board_y
        x = x.view(x.size(0), -1) # reshape tensor to 1d vectors, x.size(0) is batch size
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        pi = self.policy_head(x)
        v = self.value_head(x)
        return F.log_softmax(pi, dim=1), torch.tanh(v)


class LinearModel(nn.Module):
    def __init__(self, observation_size:tuple[int, int], action_space_size:int, config:BaseNetConfig, device:torch.device='cpu'):
        super(LinearModel, self).__init__()
        
        self.action_size = action_space_size
        self.config = config
        self.device = device
        
        observation_size = reduce(lambda x, y: x*y , observation_size, 1)
        self.l_pi = nn.Linear(observation_size, action_space_size)
        self.l_v  = nn.Linear(observation_size, 1)
        self.to(device)
    
    def forward(self, s: torch.Tensor):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(s.shape[0], -1)                                # s: batch_size x (board_x * board_y)
        pi = self.l_pi(s)
        v = self.l_v(s)
        return F.log_softmax(pi, dim=1), torch.tanh(v)
    
class MyNet(nn.Module):
    def __init__(self, observation_size: tuple[int, int], action_space_size: int, config: BaseNetConfig, device: torch.device = 'cpu'):
        super().__init__()
        self.config = config
        self.device = device

        # 卷积层
        self.conv1 = nn.Conv2d(1, config.num_channels, kernel_size=3, stride=1, padding=1)  # 卷积层，提取局部特征
        self.relu = nn.ReLU()

        # 全连接层
        input_dim = observation_size[0] * observation_size[1] * config.num_channels
        self.fc1 = nn.Linear(input_dim, config.linear_hidden[0])
        self.fc2 = nn.Linear(config.linear_hidden[0], config.linear_hidden[1])

        # 策略头和价值头
        self.policy_head = nn.Linear(config.linear_hidden[1], action_space_size)
        self.value_head = nn.Linear(config.linear_hidden[1], 1)

        self.to(device)

    def forward(self, s: torch.Tensor):
        # 输入: s (batch_size x board_x x board_y)
        s = s.unsqueeze(1)  # 添加通道维度: batch_size x 1 x board_x x board_y

        # 卷积层
        s = self.relu(self.conv1(s))

        # 展平张量以输入全连接层
        s = s.view(s.size(0), -1)

        # 全连接层
        s = self.relu(self.fc1(s))
        s = self.relu(self.fc2(s))

        # 策略头和价值头
        pi = self.policy_head(s)  # 策略头
        v = self.value_head(s)  # 价值头

        return F.log_softmax(pi, dim=1), torch.tanh(v)