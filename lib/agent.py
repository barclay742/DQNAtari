"""
Written by Barclay Zhang with assistance of the book Deep Reinforcement learning Hands on.
"""

from collections import deque,namedtuple
import random
import numpy as np
import torch
import gym
from torch.nn import MSELoss
from dqn import DQN

Experience = namedtuple('Experience',field_names=["state",'action','reward','done','new_state'])
FLOAT_TYPE = np.float32

class ExperienceBuffer():
    def __init__(self, size):

        self.buffer = deque(maxlen=size)

    def __len__(self):
        return len(self.buffer)

    def append(self, x):

        self.buffer.append(x)

    def sample(self, BATCH_SIZE):
        """
        Sample randomly from buffer
        :param BATCH_SIZE: int
        :return: states: np.array(float32) , actions : np.array(float32) ,rewards : np.array(float32) ,dones : np.array(int8), new_states : np.array(float32)
        """
        idxs = np.random.choice(len(self.buffer),BATCH_SIZE)
        states,actions,rewards,dones,new_states= zip(*[ self.buffer[idx] for idx in idxs])
        return_tuple =(np.array(states,dtype=FLOAT_TYPE), np.array(actions,dtype=FLOAT_TYPE), np.array(rewards,dtype=FLOAT_TYPE),
                       np.array(dones,dtype=np.int8), np.array(new_states,dtype=FLOAT_TYPE))
        return return_tuple

class Agent():
    def __init__(self, env, buffer):
        self.env = env
        self.ExpBuffer = buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self,policy_net,epsilon,device):
        """
        :param policy_net: DQN
        :param epsilon: Int
        :param device:  "Cuda or Cpu"
        :return: done_reward :int  if done or None else
        """
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_tensor =  torch.tensor(np.array([self.state],copy=False)).to(device)
            q_vals = policy_net(state_tensor)
            action = int(torch.argmax(q_vals,dim=1).item())

        next_state, reward, done, info = self.env.step(action)
        exp = Experience(self.state,action,reward,done,next_state)
        self.env.render()
        self.ExpBuffer.append(exp)
        self.state = next_state
        self.total_reward +=reward
        if done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calculate_loss(batch,policy_net: DQN ,target_net:DQN,device,gamma:float):
    """
    :param batch: state , action , reward ,done ,next_state
    :param policy_net:
    :param target_net:
    :param device:
    :return: MSE : float16
    """
    states , actions ,rewards ,dones, next_states = batch
    states_tensor = torch.tensor(states).to(device)
    next_states_tensor = torch.tensor(next_states).to(device)

    actions_tensor = torch.tensor(actions).to(device).to(torch.long)
    rewards_tensor = torch.tensor(rewards).to(device)
    dones_bool_mask = torch.tensor(dones).to(dtype=torch.bool)
    q_values_full = policy_net(states_tensor)

    q_values = q_values_full.gather(1,actions_tensor.unsqueeze(-1)).squeeze(1)

    next_state_values = target_net(next_states_tensor).max(1)[0]
    next_state_values[dones_bool_mask]=0.0
    next_state_values.detach()
    next_q_values = rewards_tensor + gamma * next_state_values
    return MSELoss()(q_values,next_q_values)

if __name__ == "__main__":


    pass
