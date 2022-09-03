import gym
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from Q_model import FCQ
import numpy as np


def decaying_factor(n_steps, val_max=0.99, val_min=0.01):
    coefs = np.linspace(val_min, val_max, n_steps, endpoint=True)[::-1]
    return coefs


def optimize_agent(experience_history):
    experience_matrix = np.vstack(experience_history)
    state = torch.from_numpy(experience_matrix[:, 0:4])
    next_state = torch.from_numpy(experience_matrix[:, 4:8])
    actions = torch.from_numpy(experience_matrix[:, -3]).to(dtype=torch.long)
    rewards = torch.from_numpy(experience_matrix[:, -2:-1])
    done = torch.from_numpy(experience_matrix[:, -1:])

    target_estimate = q_function(state)
    target_estimate = torch.gather(target_estimate, -1, actions.unsqueeze(-1))
    td_target = rewards + \
               gamma * q_function(next_state).max(dim=-1)[0].detach().unsqueeze(-1) * (1 - done)
    loss = criterion(target_estimate, td_target)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def RandomGame(env, batch_size):

    total_rewards_change = []
    epsilons = decaying_factor(n_steps=num_epochs)
    select_action = lambda st, Q, epsilon: torch.argmax(Q(st), dim=1)[0].item() \
        if np.random.rand() > epsilon \
        else np.random.randint(out_state)

    experience_history = []

    for episode in range(num_epochs):
        state = env.reset()
        action = select_action(torch.from_numpy(state).unsqueeze(0), q_function, epsilon)

        for t in range(500):
            next_state, reward, done, info, addit = env.step(action)
            next_action = select_action(torch.from_numpy(next_state).unsqueeze(0), q_function, epsilon)

            #is_truncated = info

            experience_history.append(np.asarray([*list(state), *list(next_state), action, reward, done], dtype=np.float32))
            if len(experience_history) >= batch_size:
                optimize_agent(experience_history)
                experience_history = []

            #td_target = reward + \
            #            gamma * q_function(torch.from_numpy(next_state).unsqueeze(0)).max().detach() * (not done)

            state, action = next_state, next_action

            if done:
                break

        if episode % 10 == 0:
            rewards = []
            for s in range(10):
                state = env.reset()
                r = 0
                for t in range(500):
                    action = select_action(torch.from_numpy(state).unsqueeze(0), q_function, 0)
                    state, reward, done, info, addit = env.step(action)
                    r += reward

                    if done:
                        break
                rewards.append(r)
            total_rewards_change.append(sum(rewards) / len(rewards))
            print(episode, total_rewards_change[-1])

    return total_rewards_change


if __name__ == "__main__":
    q_function = FCQ(in_state, out_state, hidden_states)
    optimizer = optim.RMSprop(q_function.parameters(), lr=lr)
    dynamics = RandomGame(env, batch_size)

    print(dynamics)
