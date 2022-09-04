import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from config import *
from Q_model import FCQ
import numpy as np
from utils import ExperienceQueue, StableTraining, ExponentialDecay


def RandomGame():
    best_score = -1
    total_rewards_change = []
    epsilons = ExponentialDecay()

    for episode in range(num_epochs):
        state = env.reset()
        epsilon = epsilons.current_epsilon()
        action = train_process.select_action(torch.from_numpy(state).unsqueeze(0).to(device), epsilon)

        for t in range(500):
            next_state, reward, done, info, addit = env.step(action)
            next_action = train_process.select_action(torch.from_numpy(state).unsqueeze(0).to(device),
                                                      epsilon)
            time_experience = [state, action, reward, next_state, done]
            experience_queue.put(time_experience)

            experience_sample = experience_queue.sample(batch_size)
            if experience_sample:
                train_process.optimize_model(experience_sample, device)
                train_process.update_target_model()

            state, action = next_state, next_action
            if done:
                break

        if episode % 10 == 0:
            rewards = []
            for s in range(20):
                state = env.reset()
                r = 0
                for t in range(500):
                    action = train_process.select_action(torch.from_numpy(state).unsqueeze(0).to(device), 0)
                    state, reward, done, info, addit = env.step(action)
                    r += reward

                    if done:
                        break
                rewards.append(r)
            total_rewards_change.append(sum(rewards) / len(rewards))
            print(episode, total_rewards_change[-1])

            if total_rewards_change[-1] > best_score:
                best_score = total_rewards_change[-1]
                torch.save(train_process.online_model.state_dict(), 'models/params.pt')
                print('Model parameters renewed')

        if episode % 200 == 0 and episode > 0:
            plt.plot([i for i in range(len(total_rewards_change))], total_rewards_change)
            plt.show()

    return total_rewards_change


if __name__ == "__main__":
    online_model = FCQ(in_state, out_state, hidden_states).to(device)
    optimizer = optim.RMSprop(online_model.parameters(), lr=lr)

    target_model = FCQ(in_state, out_state, hidden_states).to(device)
    for online_params, target_params in zip(online_model.parameters(), target_model.parameters()):
        target_params.data.copy_(online_params.data)

    train_process = StableTraining(online_model, target_model, criterion, optimizer, stable_target_episodes)
    experience_queue = ExperienceQueue(min_que, max_que)
    dynamics = RandomGame()
    print(dynamics)
