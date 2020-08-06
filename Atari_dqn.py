"""
Written by Barclay Zhang with assistance of the book Deep Reinforcement learning Hands on.
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
import time
from collections import deque,namedtuple
import argparse

import os
import sys
sys.path.append(".\lib")

from wrappers import make_env
from dqn import DQN


MEMORY_REPLAY_SIZE = 10_000
MEMORY_REPLAY_START_SIZE = 10_000
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY_LAST_FRAME = 150_000
GAMMA = 0.99
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1_000
TRAINING_STEP_LIMIT = 2_000_000
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"

Experience = namedtuple('Experience',field_names=["state",'action','reward','done','new_state'])
FLOAT_TYPE = np.float32

class ExperienceBuffer:
    """
    Experience buffer class with a deque as data structure
    Has a len, append and sample methods.
    """
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
        idxs = np.random.choice(len(self.buffer),BATCH_SIZE,replace=False)

        states,actions,rewards,dones,new_states= zip(*[ self.buffer[idx] for idx in idxs])

        return np.array(states), np.array(actions), \
                       np.array(rewards,dtype=FLOAT_TYPE),np.array(dones,dtype=np.int8), \
                       np.array(new_states)


class Agent:
    def __init__(self, env, buffer,render=False):
        self.env = env
        self.ExpBuffer = buffer
        self._reset()
        self.render = render

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self,policy_net  :DQN ,epsilon = 0.0 ,device=torch.device("cuda")):
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
        if self.render:
            self.env.render()
        self.ExpBuffer.append(exp)
        self.state = next_state
        self.total_reward +=reward
        if done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calculate_loss(batch,policy_net: DQN ,target_net:DQN,device):
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
    next_q_values = rewards_tensor + GAMMA * next_state_values
    return nn.MSELoss()(q_values,next_q_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--env",default="PongNoFrameskip-v4",help="Environment name")
    parser.add_argument("-bcsize","--batch_size",default=BATCH_SIZE)
    parser.add_argument("-r","--render",default=False,action="store_true",help="Render")
    parser.add_argument("-l","--load_model",default=False,help="Load model path")
    parser.add_argument("-ep_start", "--epsilon_start", default=EPSILON_START, help="Epsilon start")
    args = parser.parse_args()

    env = make_env(args.env)

    #Save directory
    save_directory = "Agents/"+args.env+"_agents"
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    #Agent
    agent = Agent(buffer=ExperienceBuffer(MEMORY_REPLAY_SIZE),env=env,render=args.render)

    #Network Creation
    n_actions = env.action_space.n
    observation_shape = env.observation_space.shape
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    policy_net = DQN(input_shape=observation_shape,n_actions=n_actions).to(device=device)
    target_net = DQN(input_shape=observation_shape,n_actions=n_actions).to(device=device)

    if args.load_model:
        state = torch.load(args.load_model, map_location=lambda stg, _: stg)
        policy_net.load_state_dict(state)
        target_net.load_state_dict(state)

    #Optimiser
    optimiser = optim.Adam(policy_net.parameters(),lr=LEARNING_RATE)

    # Setup
    EPSILON_START = int(args.epsilon_start)

    #Stats and recording
    writer = SummaryWriter(comment="-"+DEFAULT_ENV_NAME)
    total_rewards = []

    frame_last_ep = 0
    time_last_ep =time.time()

    best_m_reward = None

    for frame_idx in range(0,TRAINING_STEP_LIMIT):
        epsilon = max(EPSILON_FINAL,EPSILON_START-frame_idx/EPSILON_DECAY_LAST_FRAME)
        done_reward = agent.play_step(policy_net=policy_net,epsilon=epsilon,device=device)

        if done_reward:
            total_rewards.append(done_reward)
            frame_speed = (frame_idx-frame_last_ep)/(time.time()-time_last_ep)
            frame_last_ep = frame_idx
            time_last_ep = time.time()
            mean_reward = np.mean(total_rewards[-100:])


            print(f"{frame_idx} steps, {len(total_rewards)} games, avg reward {mean_reward:.2f}, epsilon {epsilon:.2f}, at speed {frame_speed:.2f} f/s")

            #TensorBoard update
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", frame_speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", done_reward, frame_idx)

            # Saving model and updating best reward
            if best_m_reward is None or best_m_reward < mean_reward:
                torch.save(policy_net.state_dict(), save_directory+"/"+args.env +"-best_%.0f.dat" % mean_reward)

                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.2f} -> {mean_reward:.2f}")
                best_m_reward = mean_reward


        if len(agent.ExpBuffer) < MEMORY_REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(policy_net.state_dict())

        optimiser.zero_grad()
        batch = agent.ExpBuffer.sample(BATCH_SIZE)
        loss = calculate_loss(batch,policy_net,target_net,device)
        loss.backward()
        optimiser.step()
