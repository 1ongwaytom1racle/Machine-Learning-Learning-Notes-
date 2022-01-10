import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import matplotlib.pyplot as plt

from keras.models import Sequential
from tensorflow.keras.layers import Dense
from Game import Game

def build_net():
    eval_net = Sequential([
        Dense(3,activation='relu',input_shape = (2,),name='el1'),
        Dense(len(env.actions),name='el2')
        ])       

    target_net = tf.keras.models.clone_model(eval_net)
    return eval_net,target_net

def copy_net():
    global eval_net
    global target_net
    target_net.set_weights(eval_net.get_weights())
    return target_net

def my_loss(q_target, q_eval):
    my_loss = tf1.reduce_mean(tf1.squared_difference(q_target, q_eval))
    return my_loss

def learn():
    global target_net
    global eval_net
    global batch_memory
    
    if learn_step_counter % replace_target_iter == 0:
        target_net = copy_net()
        print('\ntarget_params_replaced\n',end = '')
    # if learn_step_counter == 100:
    #     eval_net.save('Model/Maze/test1.h5')
    #     print('\nModel_saved\n')
        
    if memory_counter > memory_size:
        sample_index = np.random.choice(memory_size, size=batch_size)
    else:
        sample_index = np.random.choice(memory_counter, size=batch_size)
        batch_memory = memory[sample_index, :]
    
    q_next = target_net.predict(batch_memory[:, -n_features:])
    q_eval = eval_net.predict(batch_memory[:, :n_features])
        
    q_target = q_eval.copy()

    batch_index = np.arange(batch_size, dtype=np.int32)
    eval_act_index = batch_memory[:, n_features].astype(int)
    reward = batch_memory[:, n_features + 1]

    q_target[batch_index, eval_act_index] = reward + env.gamma * np.max(q_next, axis=1) 
    
    feed_x = batch_memory[:, :n_features]
    cost = eval_net.fit(feed_x,q_target,epochs=1,verbose=0)
    cost_his.append(cost)
    
def feature(state):
    y = [state,state]
    return np.array([y]).reshape(1,-1)

def choose_action(observation,learn_step_counter):

        if np.random.uniform() < (0.5 + 0.02*learn_step_counter):
            # forward feed the observation and get q value for every actions
            actions_value = eval_net(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, n_actions)
        if observation.all() == 0:
            action = 1
        return action
def store_transition(s, a, r, s_):
    global memory_counter 
    transition = np.hstack((s, np.array([a, r]).reshape(1,-1), s_))
    # replace the old memory with new memory
    index = memory_counter % memory_size
    memory[index, :] = transition
    memory_counter += 1    
    
env = Game()
replace_target_iter = 50
learn_step_counter = 0
memory_counter = 0
memory_size = 50
batch_size = 16
n_features = 2
n_actions = len(env.actions)
cost_his = []
memory = np.zeros((memory_size, n_features * 2 + 2)) 
batch_memory =  np.zeros((batch_size, n_features * 2 + 2)) 
step_his = []        
episode = 1
eval_net,target_net = build_net()
eval_net.compile(loss=my_loss,optimizer='RMSprop') 

# aa = np.array(eval_net(feature(0)))
for i in range(episode):
    current_state = 0
    total_steps = 0  

    while True:
        current_action = choose_action(feature(current_state),learn_step_counter)
        done,reward,next_state = env.get_next_state(current_state, current_action)
        next_action = choose_action(feature(next_state),learn_step_counter)
        store_transition(feature(current_state),current_action,reward,feature(next_state))
        if memory_counter > memory_size:
            learn()
            learn_step_counter += 1
        
        current_state = next_state

        env.update_env(current_state) # 环境相关
        if done:
            break
        total_steps += 1          # 环境相关
    step_his.append(total_steps)


for i in range(episode):
    print('\rEpisode {}: total_steps = {}'.format(i, step_his[i])) 


plt.plot(np.arange(episode),step_his)
plt.xlabel('episode')
plt.ylabel('steps')