from Game import Game
import numpy as np
import matplotlib.pyplot as plt


env = Game()


current_state = 0
total_steps = 0           

a_his = []
step_his = []
Episode_his =[]
episodes = 10
for i in range(episodes):
    current_state = 0
    total_steps = 0           # 环境相关
    while True:
        current_action = env.epsilon_greedy(current_state,i)
        a_his.append(current_action)
        done,reward,next_state = env.get_next_state(current_state, current_action)
        next_state_q_values = env.q_table[next_state, env.get_valid_actions(next_state)]
        
        env.q_table[current_state, current_action] += (
        env.alpha * (reward
        + env.gamma * next_state_q_values.max()
        - env.q_table[current_state, current_action]))
        
        current_state = next_state

        env.update_env(current_state) 
        if done:
            break
        total_steps += 1          
    step_his.append(total_steps)
    

for i in range(episodes):
    print('\rEpisode {}: total_steps = {}'.format(i, step_his[i])) 


print('\nq_table:')
print(env.q_table)


plt.plot(np.arange(episodes),step_his)
plt.xlabel('episode')
plt.ylabel('steps')
