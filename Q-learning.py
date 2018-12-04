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
from collections import namedtuple
from PIL import Image

# hyperparameters
W = 4 # number of hidden layer neurons, not used though
update_period = 10 # every how many episodes to update target network?
sample_size = 100 # the size of each training set
learning_rate = 1e-4
gamma = 0.999 # discount factor for reward
decay_rate = 0.95 # decay factor 
resume = False # resume from previous checkpoint?
render = True
np.random.seed(2333)
exploration_rate=0.9


env = gym.make('CartPole-v0')

resize = T.Compose([T.ToPILImage(),
                    #T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
#preprocess image
def getimage(env):
    
    screen=env.render(mode='rgb_array').transpose(2,0,1)
    screen=screen[0:1, 250:450:8, 100:300:4]
    screen[(screen<255) & (screen>0)]=10 #value doesn't matter
    screen[screen!=10]=0
    screen=torch.from_numpy(np.ascontiguousarray(screen))
    return resize(screen)

# Define model:
class model(nn.Module):
#TODO: transform the datatype
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5, stride=2)#(400-5-1)/2+1=198, (600-5-1)/2+1=298
        self.conv2 = nn.Conv2d(1, 1, kernel_size=5, stride=2)#(198-5-1)/2+1=97, (298-5-1)/2+1=147
        self.fc = nn.Linear(20, 2) #Predicts Q for 2 actions
    def forward(self, x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
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
    def discount(self):
        self.memory[-1]=self.memory[-1]._replace(reward = self.memory[-1].reward * len(self.memory))
        for i in range(1, len(self.memory)):
            j=len(self.memory)-i-1
            self.memory[j]=self.memory[j]._replace(reward = len(self.memory) * self.memory[j].reward + self.memory[j+1].reward * gamma)
    def sample(self, sample_size):
        self.sample_size=min(sample_size, len(self.memory))
        train_set = random.sample(self.memory,  self.sample_size)
        train_set = trial(*zip(*train_set))
        return torch.stack(train_set.state), torch.stack(train_set.action), torch.stack(train_set.next_state), torch.stack(train_set.reward)




policyNN=model()  #create two instances of model
targetNN=model()
if resume:
    policyNN.load_state_dict(pickle.load(open('save_Qlearning.p', 'rb')))
targetNN.load_state_dict(policyNN.state_dict())
targetNN.eval()

optimizer = optim.RMSprop(policyNN.parameters())#, lr=learning_rate, weight_decay=decay_rate)
memory = ReplayMemory() # stores memory in a batch
n_episode=0
n_steps=0
observation = env.reset()
state=getimage(env)
this_input=state
import math
while True: #Keep playing the game until I stop you
    rdn=np.random.uniform(0,1)
    if rdn < exploration_rate * math.exp(- n_episode/100):
        action = torch.tensor([np.random.randint(2)]) #take a random action in rare case
    else: 
        action = policyNN(this_input.unsqueeze(0)).max(1)[1].view(1) # otherwise take action with best expected rewards
    n_steps+=1
    observation, reward, done, info = env.step(action.item()) # take a random action
    last_input = this_input
    this_input = getimage(env) - state
    state = getimage(env)
    reward=torch.tensor([reward])
    memory.push(last_input, action, this_input, reward)
    if done: # finished a game
        n_episode+=1
        print('{} episodes finished with duration {}'.format(n_episode, n_steps))
        n_steps=0
        observation = env.reset() # reintialize the env
        state = getimage(env)
        this_input = state
        if len(memory) > sample_size: # let's start training
            state_batch, action_batch, next_state_batch, reward_batch = memory.sample(sample_size)
            current_Q_batch = policyNN(state_batch).gather(1,action_batch)
            expectedQ_batch = targetNN(next_state_batch).max(1)[0].detach().unsqueeze(1) * gamma + reward_batch
            loss = F.smooth_l1_loss(current_Q_batch, expectedQ_batch) # Hubber loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('training loss = {}'.format(loss.item()))
        if n_episode % update_period == 0: # time to update target net
            targetNN.load_state_dict(policyNN.state_dict())
        if n_episode % 10*update_period ==0: # save current model in case you want to resume
            pickle.dump(policyNN.state_dict(), open('save_Qlearning.p', 'wb'))




