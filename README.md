# CartPoleEnvironment
Introduction into Deep Reinforcement Learning with DQN algorithm project

## Project files
- **Q_model.py**: agent action-value function q(s, a) approximation neural network
- **config.py**: project configuration
- **main.py**: training loop
- **utils.py**: additional utils for training like experience monitoring queue, exponential decay for greedy learning and agent version controll
function for training stability

## Results
Agent managed to perform 500 milliseconds reward 20 times in a row, thus environment may be considered resolved. 

## **Studying resources**
**Groking deep reinforcement leraning book (Chapters 8-9)**: https://www.oreilly.com/library/view/grokking-deep-reinforcement/9781617295454/
