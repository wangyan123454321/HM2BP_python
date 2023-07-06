import torch
import torch.nn as nn
from torch.nn import functional as ff
from torch.autograd import Function as gf
import math
from . import functions as fun

class mm_firing_counter(gf):
    @staticmethod
    def forward(ctx, input):
        """
        脉冲计数器的前向传播函数，用于统计脉冲发射个数
        @params:
            input: 来自输出层的脉冲，形状为[time_steps, batch_size, input_shape]
        @return:
            output: 统计过后的脉冲发射总数，形状为[batch_size, output_shape = input_shape]
        """
        output = torch.sum(input, dim = 0) # [batch_size, output_shape]
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播时，直接将梯度传播至前面的网络中
        @params:
            grad_output: 输出层的梯度，形状为[batch_size, output_shape]
        @return:
            grad_input: 输入层的梯度，形状为[batch_size, input_shape = output_shape]
        """
        grad_input = grad_output
        return grad_input

class FiringCounter(nn.Module):
    """
    脉冲计数层，在反传的时候不会在时间上展开。
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.mean(x, dim = 0)