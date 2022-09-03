import gym
import torch.nn as nn

num_epochs = 3000
gamma = 1.
env = gym.make('CartPole-v1', render_mode='human', new_step_api=True)
in_state = 4
out_state = 2
hidden_states = (32, 16)
criterion = nn.MSELoss()
batch_size = 256
lr = 0.00005
epsilon = 0.5
