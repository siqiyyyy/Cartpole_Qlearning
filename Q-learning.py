import gym
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable
import matplotlib.pyplot as plt
from collections import namedtuple
from PIL import Image

# hyperparameters
DEBUG = True
W = 1 # number of hidden layer neurons, not used though
update_period = 10 # every how many episodes to update target network?
sample_size = 100 # the size of each training set
learning_rate = 1e-4
gamma = 0.999 # discount factor for reward
decay_rate = 0.95 # decay factor 
resume = False # resume from previous checkpoint?
render = True
np.random.seed(2333)
exploration_rate=0.9

plt.ion()
if DEBUG:
    plt.figure()
    plt.show()

env = gym.make('CartPole-v0')

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.LANCZOS),
                    T.ToTensor()])
#preprocess image
def getimage(env):
    
    screen=env.render(mode='rgb_array').transpose(2,0,1)
    screen=screen[:, 200:320, 150:450]
    #screen[(screen<250) & (screen>5)]=10 #value doesn't matter
    #screen[screen==1]=0
    screen=torch.from_numpy(np.ascontiguousarray(screen))
    return resize(screen)

# Define model:
class model(nn.Module):
#TODO: transform the datatype
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3, W, kernel_size=5, stride=2)
        self.bn1=nn.BatchNorm2d(W)
        self.conv2 = nn.Conv2d(W, W, kernel_size=5, stride=2)
        self.bn2=nn.BatchNorm2d(W)
        self.fc = nn.Linear(154*W, 2) #Predicts Q for 2 actions
    def forward(self, x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x)))
        x=self.fc(x.view(x.size(0),-1))
        return x

#Replay memory
trial=namedtuple('trial', ('state','action','next_state','reward'))
class ReplayMemory(object):
    def __init__(self, capacity=0): #capacity=0 for infinite capacity
        self.capacity=capacity
        self.memory=[]
    def push(self, *args):
        #if self.capacity==0 or len(self.memory) < self.capacity:
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




policyNN=model()  #create two instances of model
targetNN=model()
if resume:
    policyNN.load_state_dict(pickle.load(open('save_Qlearning.p', 'rb')))
targetNN.load_state_dict(policyNN.state_dict())
targetNN.eval()
policyNN.train()

optimizer = optim.RMSprop(policyNN.parameters())#, lr=learning_rate, weight_decay=decay_rate)
memory = ReplayMemory() # stores memory in a batch
n_episode=0
n_steps=0
observation = env.reset()
screen=getimage(env)
this_state=screen
import math
while True: #Keep playing the game until I stop you
    rdn=np.random.uniform(0,1)
    if rdn < exploration_rate * math.exp(- n_episode/100):
        action = torch.tensor([np.random.randint(2)]) #take a random action in rare case
    else: 
        action = policyNN(this_state.unsqueeze(0)).max(1)[1].view(1) # otherwise take action with best expected rewards
    if DEBUG: print('taking action '+str(action)+' and expected future rewards {}'.format([i.item() for i in policyNN(this_state.unsqueeze(0))[0]]))
    n_steps+=1
    observation, reward, done, info = env.step(action.item()) # take a random action
    last_state = this_state
    this_state = getimage(env) # - screen
    screen = getimage(env)
    if DEBUG:
        plt.clf()
        plt.imshow(this_state.squeeze().numpy()[0,:,:])
        plt.pause(0.001)
    reward=torch.tensor([reward])
    memory.push(last_state, action, this_state, reward)
    if done: # finished a game
        memory.push(this_state, torch.tensor([np.random.randint(2)]), None, torch.tensor([0.0])) #learn when is gameover
        n_episode+=1
        print('{} episodes finished with duration {}'.format(n_episode, n_steps))
        n_steps=0
        observation = env.reset() # reintialize the env
        screen = getimage(env)
        this_state = screen
        if len(memory) > sample_size: # let's start training
            state_batch, action_batch, next_state_batch, reward_batch = memory.sample(sample_size)
            current_Q_batch = policyNN(torch.stack(state_batch)).gather(1,torch.stack(action_batch))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,next_state_batch)))
            non_final_next_states = torch.cat([s.unsqueeze(0) for s in next_state_batch if s is not None])
            expectedQ_batch = torch.zeros(sample_size).unsqueeze(1)
            expectedQ_batch[non_final_mask] = targetNN(non_final_next_states).max(1)[0].detach().unsqueeze(1) * gamma + torch.stack(reward_batch)[non_final_mask]
            loss = F.smooth_l1_loss(current_Q_batch, expectedQ_batch) # Hubber loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('training loss = {}'.format(loss.item()))
        if n_episode % update_period == 0: # time to update target net
            targetNN.load_state_dict(policyNN.state_dict())
        if n_episode % 10*update_period ==0: # save current model in case you want to resume
            pickle.dump(policyNN.state_dict(), open('save_Qlearning.p', 'wb'))




