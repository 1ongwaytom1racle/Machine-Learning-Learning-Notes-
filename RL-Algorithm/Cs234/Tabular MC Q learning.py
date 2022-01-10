from Game import Game
import numpy as np
import matplotlib.pyplot as plt
env = Game()

current_state = 0
total_steps = 0         
a_his = []
step_his = []
reward_his =[]
episode = 10
for i in range(episode):
    memory =[]
    current_state = 0
    total_steps = 0         
    while True:
        current_action = env.epsilon_greedy(current_state,episode)
        a_his.append(current_action)
        done,reward,next_state = env.get_next_state(current_state, current_action)
        next_state_q_values = env.q_table[next_state, env.get_valid_actions(next_state)]
        
        memory.append([current_state,current_action,reward])
        
        current_state = next_state

        env.update_env(current_state) 
        if done:
            break
        total_steps += 1          
    step_his.append(total_steps)
    Git = np.zeros([total_steps + 1,1])
    G = np.zeros([len(env.states),4])#后两列为计数列
    for i in range(total_steps + 1):
        for j in range(i,total_steps + 1):
            Git[i] += np.power(env.gamma,j-i)*memory[j][2]
            
    for i in range(total_steps + 1):
        G[memory[i][0],memory[i][1]] += Git[i]
        G[memory[i][0],2+ memory[i][1]] += 1    #同一行的 第2+action列即是 这个S-A出现的次数
    
    for i in range(len(env.states)):
        for j in range(len(env.actions)):
            if G[i,j+2] != 0:
                env.q_table[i,j] = G[i,j]/G[i,j+2]
            else:
                env.q_table[i,j] = 0

for i in range(episode):
    print('\rEpisode {}: total_steps = {}'.format(i, step_his[i])) 


print('\nq_table:')
print(env.q_table)

plt.plot(np.arange(episode),step_his)
plt.xlabel('episode')
plt.ylabel('steps')



