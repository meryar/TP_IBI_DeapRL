from Agent import Agent
import gym
import time
import random
import sys
import torch
import matplotlib.pyplot as plt



#####################  hyper parameters  ####################

Env_name = 'CartPole-v1'
Action_nb = 2
Env_dim = 4
nb_episode = 150
Epsilon = 0.1
Batch_size = 32


#############################################################

if __name__ == '__main__':
    counter = 0
    env = gym.make(Env_name)
    env.seed(0)

    file = "model/model"
    if len(sys.argv) > 1:
        file += str(sys.argv[1])
    file += ".pt"

    ag = Agent(Env_dim, Action_nb)

    t1 = time.time()
    for i in range(nb_episode):
        s = env.reset()
        ep_reward = 0
        while True:
            if random.random() < Epsilon:
                a = env.action_space.sample()
            else:
                output = ag.eval_nn(torch.FloatTensor(s))
                a = int(torch.argmax(output))

            sn, r, done, _ = env.step(a)
            if done :
                r = -10
            ag.store_transition((s, a, sn, r, done))
            ag.optimize_model(file)
            s = sn
            ep_reward += r
            if done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), )
                break
    print('Running time: ', time.time() - t1)
    rewards = []
    for i in range(200):
        s = env.reset()
        ep_reward = 0
        while True:

            output = ag.eval_nn(torch.FloatTensor(s))
            a = int(torch.argmax(output))
            s_, r, done, _ = env.step(a)
            s = s_
            ep_reward += r
            if done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), )
                break
        rewards.append(ep_reward)
    plt.plot(rewards)
    plt.show()