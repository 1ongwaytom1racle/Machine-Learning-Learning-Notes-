import numpy as np
import random
import time
class Game(object):
    def __init__(self):
        self.epsilon = 0.5   # 贪婪度 greedy
        self.alpha = 0.01     # 学习率
        self.gamma = 0.9     # 奖励递减值
        self.states = np.arange(5)           # 状态集。从0到5
        self.actions = [0, 1] #0向左，1向右# 动作集。也可添加动作'none'，表示停留
        self.rewards = [0,0,0,0,1]
        self.q_table = np.zeros((len(self.states),len(self.actions)))
        self.v_table = np.zeros((len(self.states),1))
        self.current_state = 0
        self.env = list('----T')
    def update_env(self,current_state):        
        if current_state != self.states[-1]:
            self.env[current_state] = 'o'
        print('\r{}'.format(''.join(self.env)), end='')
        time.sleep(0.1)
        self.env = list('----T')
      
    def get_next_state(self,current_state,action):         
         # l,r,n = -1,+1,0
        if action == 1 : # 除非最后一个状态（位置），向右就+1
            next_state = current_state + 1
        elif action == 0 and current_state != self.states[0]: # 除非最前一个状态（位置），向左就-1
            next_state = current_state -1
        else:
            next_state = current_state
            
        if next_state == self.states[-1]:
            done = True
            reward = 1
        else:
            done = False
            reward = 0
        return done,reward,next_state  
    
    def get_valid_actions(self,current_state):
        
        self.valid_actions = self.actions
        if current_state == self.states[0]:              # 最前一个状态（位置），则
            self.valid_actions = [1]  # 不能向左
        
        return self.valid_actions
    def get_random(self):
        self.r = random.uniform(0, 1)
        return self.r
        
    def epsilon_greedy(self,current_state,i):
        r = self.get_random()
        valid_actions = self.get_valid_actions(current_state)
        if current_state == self.states[-1]:
            current_action = 0    
        if (r > self.epsilon + 0.008*i) or ((self.q_table[current_state]).all() == 0):  # 探索
                # current_action = random.choice(self.get_valid_actions(current_state))
            current_action = random.choice(valid_actions)                
        else:
            current_action = np.argmax(self.q_table[current_state]) # 利用（贪婪）
        return current_action
    
    def epsilon_greedy_VFA(self,current_state,feature,W,i):
        r = self.get_random()
        valid_actions = self.get_valid_actions(current_state)   
        if (r > (self.epsilon + 0.02*i)) :  # 探索
            current_action = random.choice(valid_actions)                
        else:
            current_action = np.argmax([np.dot(feature,W)]) # 利用（贪婪）
        if current_state == self.states[0] :
            current_action = 1    
        if current_state == self.states[-1]:
            current_action = 0             
        return current_action                    

    