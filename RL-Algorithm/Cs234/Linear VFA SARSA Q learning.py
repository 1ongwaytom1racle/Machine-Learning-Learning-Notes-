from Game import Game
import numpy as np
import matplotlib.pyplot as plt

env = Game()

a_his = []
step_his = []
Episode_his =[]
initial_W = 0.1*np.ones([2,2])
def feature(state):
    y = [state,state]
    return np.array([y]).reshape(1,-1)
bb = np.dot(feature(1),initial_W)


W = initial_W
episode = 100
for i in range(episode):
    memory =[]
    current_state = 0
    total_steps = 0           # 环境相关
    while True:
        current_action = env.epsilon_greedy_VFA(current_state,feature(current_state),W,i)
        a_his.append(current_action)
        done,reward,next_state = env.get_next_state(current_state, current_action)
        next_action = env.epsilon_greedy_VFA(next_state,feature(next_state),W,i)
        delta_W = env.alpha*(reward + 
                              env.gamma*(np.dot(feature(next_state),W)[0,next_action]) 
                              - np.dot(feature(current_state),W)[0,current_action])*feature(current_state)
        memory.append([current_state,current_action,reward])
        
        W[0,memory[-1][1]] += delta_W[:,0] 
        W[1,memory[-1][1]] += delta_W[:,1]

        
        current_state = next_state

        env.update_env(current_state) # 环境相关
        if done:
            break
        total_steps += 1          # 环境相关
    step_his.append(total_steps)


for i in range(episode):
    print('\rEpisode {}: total_steps = {}'.format(i, step_his[i])) 
print('\nW:',W)

plt.plot(np.arange(episode),step_his)
plt.xlabel('episode')
plt.ylabel('steps')


