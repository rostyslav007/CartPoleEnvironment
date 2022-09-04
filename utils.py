import torch
import torch.nn as nn
import numpy as np
from random import sample
from config import out_state
import copy


class StableTraining:
    def __init__(self, online_model, target_model, criterion, optimizer, stable_target_episodes=15, gamma=1.):
        self.online_model = online_model
        self.target_model = target_model
        self.gamma = gamma
        self.criterion = criterion
        self.optimizer = optimizer
        self.renew_count = stable_target_episodes
        self.counter = 0

    def select_action(self, state, epsilon):
        return torch.argmax(self.online_model(state), dim=1)[0].item() \
            if np.random.rand() > epsilon else np.random.randint(out_state)

    def optimize_model(self, experience, device):
        states = torch.from_numpy(np.vstack([e[0] for e in experience])).to(device=device, dtype=torch.float32)
        actions = torch.from_numpy(np.asarray([e[1] for e in experience], dtype=np.int64)).to(device)
        rewards = torch.from_numpy(np.asarray([e[2] for e in experience], dtype=np.float32)).to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experience])).to(device=device, dtype=torch.float32)
        dones = torch.from_numpy(np.asarray([e[4] for e in experience], dtype=np.float32)).to(device)

        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp.max(1)[0]
        max_a_q_sp *= (1 - dones)

        target_q_sa = rewards + self.gamma * max_a_q_sp
        q_sa = self.online_model(states).gather(1, actions.unsqueeze(-1))

        value_loss = self.criterion(torch.flatten(q_sa), target_q_sa)
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.counter += 1
        if self.counter >= self.renew_count:
            for online, target in zip(self.online_model.parameters(), self.target_model.parameters()):
                target.data.copy_(online.data)
            self.counter = 0
            print('Target model version renewed')


class ExperienceQueue:
    def __init__(self, min_count, max_count):
        self.min_count = min_count
        self.max_count = max_count
        self.queue = []

    def put(self, experience):
        self.queue.append(experience)
        if len(self.queue) > self.max_count:
            oldest_sample = self.queue.pop(0)

    def sample(self, batch_size=256):
        experience = sample(self.queue, batch_size) if batch_size < len(self.queue) else []
        return experience

    def clear(self):
        self.queue = []

    class EGreedyExpStrategy():
        def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
            self.epsilon = init_epsilon
            self.init_epsilon = init_epsilon
            self.decay_steps = decay_steps
            self.min_epsilon = min_epsilon
            self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
            self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
            self.t = 0
            self.exploratory_action_taken = None

        def _epsilon_update(self):
            self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
            self.t += 1
            return self.epsilon

        def select_action(self, model, state):
            self.exploratory_action_taken = False
            with torch.no_grad():
                q_values = model(state).detach().cpu().data.numpy().squeeze()

            if np.random.rand() > self.epsilon:
                action = np.argmax(q_values)
            else:
                action = np.random.randint(len(q_values))

            self._epsilon_update()
            self.exploratory_action_taken = action != np.argmax(q_values)
            return action


class ExponentialDecay():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None

    def current_epsilon(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

