"""
Written by Barclay Zhang with assistance of the book Deep Reinforcement learning Hands on.
"""


import torch
import torch.nn as nn
import numpy as np

import wrappers

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self,input_shape):
        o = self.conv(torch.zeros((1,*input_shape)))
        return int(np.prod(o.size()))

    def forward(self,x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size()[0] , -1)
        return self.fc(conv_out)


if __name__ == "__main__":
    env = wrappers.make_env("Pong-v0")
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n
    print("input shape",input_shape)
    print("n_actions",n_actions)
    net = DQN(input_shape,n_actions)

    obs = env.reset()
    obs1 =  env.reset()
    t = torch.tensor([obs,obs1])

    print("x.shape",t.shape)

    net.forward(t)
    a = torch.zeros((7,7))
    print(a.view(-1))