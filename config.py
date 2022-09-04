import gym
import torch
import torch.nn as nn

num_epochs = 5000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gamma = 1.
env = gym.make('CartPole-v1', render_mode='human', new_step_api=True)
in_state = 4
out_state = 2
hidden_states = (512, 128)
criterion = nn.MSELoss()
batch_size = 512
lr = 0.0005
min_que = 1000
max_que = 3000
stable_target_episodes = 25
