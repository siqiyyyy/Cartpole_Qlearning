import gym
import numpy as np
import pickle
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable
import matplotlib.pyplot as plt
from collections import namedtuple

# hyperparameters
DEBUG = True # will dump more info if set True
W = 10 # number of nodes at the hidden layer
sample_size = 50 # the size of each training batch
learning_rate = 1e-4 
gamma = 0.999 # discount factor for reward
decay_rate = 0.95 # decay factor 
resume = False # resume from previous checkpoint?
render = True
update_period = 5
np.random.seed(2333)
initial_exploration_rate=0.9
final_exploration_rate=0.1
exploration_rate_decay = 0.02

env = gym.make('CartPole-v0')

if DEBUG:
    plt.ion()
    fig=plt.gcf()
    plt.clf()
    fig.show()

# Define model:
class model(nn.Module):
#TODO: transform the datatype
    def __init__(self):
        super(model, self).__init__()
        self.fc1 = nn.Linear(4, W)
        self.fc2 = nn.Linear(W, W)
        self.fc3 = nn.Linear(W, 2)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

#Replay memory
trial=namedtuple('trial', ('state','action','next_state','reward'))
class ReplayMemory(object):
    def __init__(self): 
        self.memory=[]
    def push(self, *args):
        self.memory.append(trial(*args))
    def append(self, extra):
        self.memory=self.memory + extra.memory
    def __len__(self):
        return len(self.memory)
    def sample(self, sample_size):
        self.sample_size=min(sample_size, len(self.memory))
        train_set = random.sample(self.memory,  self.sample_size)
        train_set = trial(*zip(*train_set))
        return train_set.state, train_set.action, train_set.next_state, train_set.reward
memory=ReplayMemory() # All playing history will be stored here

# initialize the Deep Q leaning network (not deep at all)
policyNN = model()
targetNN = model()
if resume:
    policyNN.load_state_dict(pickle.load(open('save_1.p', 'rb')))

targetNN.load_state_dict(policyNN.state_dict())
optimizer = optim.RMSprop(policyNN.parameters())#, lr=learning_rate, weight_decay=decay_rate)
n_episode=0
n_steps=0
observation = env.reset()
if render:
    img=env.render('mode=rgb_array')
this_state = torch.FloatTensor([observation])
loss_track = []

def take_action(state):
    toss=np.random.uniform(0,1)
    if toss < final_exploration_rate + (initial_exploration_rate - final_exploration_rate) * math.exp(- n_episode * exploration_rate_decay):
        action = torch.tensor([np.random.randint(2)]) #take a random action in rare case
    else: 
        action = policyNN(state).max(1)[1].view(1) # otherwise take action with best expected rewards
    if DEBUG: 
        print('taking action '+str(action)+' and expected future rewards {}'.format(policyNN(state).detach().numpy()))
    return action

def train():
    if len(memory) < sample_size:
        return
    state_batch, action_batch, next_state_batch, reward_batch = memory.sample(sample_size)
    current_Q_batch = policyNN(torch.stack(state_batch)).squeeze().gather(1,torch.stack(action_batch))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,next_state_batch)))
    non_final_next_states = torch.cat([s.unsqueeze(0) for s in next_state_batch if s is not None])
    expectedQ_batch = torch.zeros(sample_size).unsqueeze(1)
    expectedQ_batch[non_final_mask] = targetNN(non_final_next_states).max(2)[0].detach()* gamma 
    expectedQ_batch = expectedQ_batch + torch.stack(reward_batch)
    loss = F.smooth_l1_loss(current_Q_batch, expectedQ_batch) # Hubber loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('training loss = {}'.format(loss.item()))
    if n_episode % update_period == 0: # time to update target net
        targetNN.load_state_dict(policyNN.state_dict())
    if n_episode % 10*update_period ==0: # save current model in case you want to resume
        pickle.dump(policyNN.state_dict(), open('save_Qlearning.p', 'wb'))
    loss_track.append(loss.item())
    if DEBUG and len(loss_track)>10:
        plt.plot(loss_track)
        plt.pause(0.1)


while True: #Keep playing the game until I stop you
    action = take_action(this_state)
    last_state = this_state
    observation, reward, done, info = env.step(action.item()) 
    n_steps+=1
    this_state = torch.FloatTensor([observation])
    reward=torch.FloatTensor([reward])
    memory.push(last_state, action, this_state, reward)
    if done: # finished a game
        memory.push(this_state, torch.tensor([np.random.randint(2)]), None, torch.tensor([0.0])) #learn when is gameover
        n_episode+=1
        print('{} episodes finished with duration {}'.format(n_episode, n_steps))
        n_steps=0
        observation = env.reset() # reintialize the env
        this_state = torch.FloatTensor([observation])
        if len(memory)>sample_size:   
            train()

