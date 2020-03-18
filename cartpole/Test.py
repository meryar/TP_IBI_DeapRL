import gym
import sys
import torch
from Agent import Agent
from RandomAgent import RandomAgent
import matplotlib.pyplot as plt

Env_name = 'CartPole-v1'
Action_nb = 2
Env_dim = 4

if __name__ == "__main__":
    print("start")
    
    file = "model/model"
    if len(sys.argv) > 1:
        file += str(sys.argv[1])
    file += ".pt"

    env = gym.make(Env_name)
    #env.seed(0)
    ob = env.reset()
    
    ag = torch.load(file)
    #ag = RandomAgent(env.action_space)

    episode_count = 10
    reward = 0
    done = False
    
    rewards = []
    
    for i in range(episode_count):
        s = env.reset()
        r = 0
        done = False
        ep_reward = 0
        while True:
            env.render()

            #output = ag.act(s,r,done)
            
            output = ag(torch.FloatTensor(s))
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
    env.close()
    
    