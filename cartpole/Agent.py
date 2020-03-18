
import torch
import torch.nn as nn
from Network import Network
from Buffer import Buffer


#####################  hyper parameters  ####################

Memory_size = 100000
Learning_rate = .001
Batch_size = 32
Gamma = 0.95     # reward discount
Refresh_gap = 1000

###############################  Agent  ####################################
class Agent():
    def __init__(self, Env_dim, Nb_action):
        self.memory = Buffer(Memory_size)
        self.eval_nn = Network(Env_dim, Nb_action)
        self.target_nn = Network(Env_dim, Nb_action)
        self.optimizer = torch.optim.Adam(self.eval_nn.parameters(),lr = Learning_rate)
        self.criterion = nn.MSELoss(reduction='sum')
        self.counter = 0
        self.target_nn.fc1 = self.eval_nn.fc1
        self.target_nn.fc2 = self.eval_nn.fc2
        self.target_nn.out = self.eval_nn.out

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.eval_nn(s)[0].detach() # ae（s）

    def getSample(self):
        return self.memory.sample(Batch_size)
        
    def optimize_model(self, file):
        if self.memory.get_nb_elements() >= Batch_size:
            batch = self.memory.sample(Batch_size)
            for s, a, s_, r, done in batch:
                qValues = (self.eval_nn(torch.tensor(s).float()))[a]
                qValues_ = self.target_nn(torch.tensor(s_).float())
                qValues_target = Gamma * torch.max(qValues_)
                JO = pow(qValues - (r + (qValues_target * (1 -done))), 2)
                loss = self.criterion(qValues, JO)
                self.optimizer.zero_grad()
                # if i != Batch_size - 1:
                #     loss.backward(retain_graph=True)
                # else:
                #     loss.backward()
                loss.backward()
                self.optimizer.step()
            self.counter += 1
            if self.counter % Refresh_gap == 0:  
                torch.save(self.eval_nn, file)
                self.target_nn.fc1 = self.eval_nn.fc1
                self.target_nn.fc2 = self.eval_nn.fc2
                self.target_nn.out = self.eval_nn.out

    def store_transition(self, value):
        self.memory.insert(value)


