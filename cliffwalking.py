import numpy as np
import random as rnd
from copy import copy, deepcopy
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt

class State:
    def __init__(self, r, c):
        self.r = r
        self.c = c

cumrew_per_episode = []
max_actions_taken = 500
cliffworld = (np.ones((5, 10))*(-1)).tolist()
cliffworld[-1][-1] = 10   # goal
cliffworld[-1][0] = -1     # start
cliffworld[-1][1:-1] = (np.ones(8)*(-100)).tolist()  # cliff
actions = ['n', 'e', 's', 'w']

Q_ = [] # look-up table for Q values! all initialized arbitraryly
sample = list(np.random.normal(0, 0.0001, 50 * 4))
for s in range(50):
    if s >= 1 + 4 * 10:
        Q_.append({'n': 0, 'e': 0, 's': 0, 'w': 0})  # terminals are initialized to zero
    else:
        Q_.append({'n': sample.pop(), 'e': sample.pop(), 's': sample.pop(), 'w': sample.pop()})  # go anywhere you like!
Q_default = deepcopy(Q_)

def clamp(pos, lim):
    return min(lim-1, max(pos, 0))

def action(state, a): # method for taking given a from s, returning its consequences (termination, reward, new state)
    row, col = state.r, state.c
    if a is 'n':
        row += -1
    elif a is 'e':
        col += 1
    elif a is 's':
        row += 1
    elif a is 'w':
        col += -1

    row = clamp(row, len(cliffworld))
    col = clamp(col, len(cliffworld[0]))
    reward = cliffworld[row][col]

    if reward == 0 or reward == -100 or reward == 10:
        return True, reward, State(row, col)
    return False, reward, State(row, col)

def epsilon_greedy(S, epsilon):
    if epsilon > rnd.uniform(0, 1):
        return rnd.choice(list(Q_[S.r * 10 + S.c].keys())) # random action
    else:
        return max(zip(Q_[S.r * 10 + S.c].values(), Q_[S.r * 10 + S.c].keys()))[1] # best action

def q_learning_update(S, A, next_S):
    alpha = 0.8
    maxQ = max(zip(Q_[next_S.r * 10 + next_S.c].values(), Q_[next_S.r * 10 + next_S.c].keys()))[0]    # get most valuable action of next state (max Q(s',a'))
    R = cliffworld[next_S.r][next_S.c]                                                  # reward acquired by arriving at s'
    Q_[S.r * 10 + S.c][A] += alpha * (R + maxQ - Q_[S.r * 10 + S.c][A])           # update rule

def SARSA_update(S, A, R, next_S, next_A):
    alpha = 0.2
    Q_[S.r * 10 + S.c][A] += alpha * (R + Q_[next_S.r * 10 + next_S.c][next_A] - Q_[S.r * 10 + S.c][A])  # update rule

def SARSA(epsilon):
    for episode in range(1000):
        terminated, cumulative_reward, episode_policy = False, 0, []
        # print('episode', episode)

        S = State(4, 0)                             # always initalize to start state (row:4,col:0)
        A = epsilon_greedy(S, epsilon)              # choose a from s using epsilon greedy policy

        while not terminated and len(episode_policy) < max_actions_taken:
            episode_policy.append(A)                # keeping track of actions, in case we wan't to check them
            terminated, R, next_S = action(S, A)    # get s' acquired from s and deterministic a
            next_A = epsilon_greedy(next_S, epsilon)         # get a' from s' using epsilon greedy policy
            cumulative_reward += R                  # add to cumulative reward of episode
            SARSA_update(S, A, R, next_S, next_A)   # SARSA update
            S = next_S
            A = next_A
        cumrew_per_episode.append(cumulative_reward)
        # print('state:', str(S.r) + ',' + str(S.c), 'action:', episode_policy[-1], 'reward:', R, 'cumrew:', cumulative_reward, 'term?:', terminated)

def Q_Learning(epsilon):
    for episode in range(1000):
        S = State(4, 0)                                     # always initalize to start state (row:4,col:0)
        terminated, cumulative_reward, episode_policy = False, 0, []
        # print('episode', episode)

        while not terminated and len(episode_policy) < max_actions_taken:
            A = epsilon_greedy(S, epsilon)                           # choose a from s using epsilon greedy policy
            terminated, R, next_S = action(S, A)            # get s' acquired from s and deterministic a
            episode_policy.append(A)                        # keeping track of actions, in case we wan't to check them
            cumulative_reward += R                          # add to cumulative reward of episode
            q_learning_update(S, A, next_S)                 # Q-learning update
            S = next_S
        # print('state:', str(S.r) + ',' + str(S.c), 'action:', episode_policy[-1], 'reward:', R, 'cumrew:', cumulative_reward, 'term?:', terminated)
        cumrew_per_episode.append(cumulative_reward)
        df2 = pd.DataFrame(Q_)

def colormap(a, name):
    dummy = deepcopy(cliffworld)
    del dummy[len(dummy)-1]
    for r in range(len(dummy)):
        for c in range(len(dummy[0])):
                dummy[r][c] = Q_[r * 10 + c][a]

    df = pd.DataFrame(dummy)

    df = np.round(df, decimals=3)
    fig, ax = plt.subplots()
    matrix = np.array(df.values)
    ax.matshow(matrix,cmap=plt.cm.Blues)
    ax.set_yticks(np.arange(4))
    ax.set_title("Q(S, A=" + name + ") values after 1000 episodes")
    ax.set_yticklabels([1, 2, 3, 4])
    ax.set_xticklabels([])

    for i in range(10):
        for j in range(4):
            c = matrix[j, i]
            ax.text(i, j, str(c), va='center', ha='center')
    plt.show()

def char_cliffworld():
    dummy = deepcopy(cliffworld)
    for r in range(len(cliffworld)):
        for c in range(len(cliffworld[0])):
            if cliffworld[r][c] == -100:
                dummy[r][c] = u"\u00D7"
            elif cliffworld[r][c] == 10:
                dummy[r][c] = 'G'
            else:
                state_actions = Q_[c + r * 10]
                a = max(zip(state_actions.values(), state_actions.keys()))[1]
                dummy[r][c] = ">" if a == 'e' else ("<" if a == 'w' else (u"\u02C4" if a == 'n' else u"\u02C5"))

    for row in dummy:
        line = ''
        for cell in row:
            line += cell + ' '
        print(line)
    return pd.DataFrame(dummy)


# SARSA(0.9)
Q_Learning(0.1)

# colormap('n', 'north/up')
# colormap('e', 'east/right')
# colormap('s', 'south/down')
# colormap('w', 'west/left')

# df0_Cliff = pd.DataFrame(cliffworld)
df0_Char = char_cliffworld()
# df0_Q = pd.DataFrame(Q_)
